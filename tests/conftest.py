"""测试夹具：完全离线的 Hugging Face 元数据、下载与运行器。"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FIXTURES = ROOT / "tests" / "fixtures"
MOCK_RUNNER = FIXTURES / "mock_llama_server.py"
FIXED_SHA = "a" * 40
REPO = "TestOrg/offline-tiny-GGUF"
VARIANT = "offline-tiny-Q4_K_M.gguf"

CONTENTS = {
    VARIANT: b"GGUF-TEST-WEIGHT-" + b"\x00" * 512,
    "README.md": b"# offline tiny\nA deterministic test model.\n",
    "LICENSE": b"Apache License 2.0 (test fixture)\n",
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FakeHfApi:
    def __init__(self):
        self.calls = {"list_models": 0, "model_info": 0}

    def list_models(self, **kwargs):
        self.calls["list_models"] += 1
        return [SimpleNamespace(id=REPO, lastModified="2026-11-10T01:00:00+00:00")]

    def model_info(self, repo_id, files_metadata=False, revision=None, token=None):
        self.calls["model_info"] += 1
        siblings = []
        for name, data in CONTENTS.items():
            siblings.append(SimpleNamespace(
                rfilename=name, size=len(data), blob_id=None,
                lfs={"size": len(data), "sha256": _sha(data), "oid_sha256": _sha(data)}))
        return SimpleNamespace(
            id=REPO, sha=FIXED_SHA, gated=False, siblings=siblings,
            cardData={"license": "apache-2.0", "pipeline_tag": "text-generation",
                      "model-index": []},
        )


class FakeFetch:
    def __init__(self):
        self.calls: list[tuple[str, str, str]] = []

    def __call__(self, repo: str, filename: str, revision: str, dest: Path) -> Path:
        self.calls.append((repo, filename, revision))
        if filename not in CONTENTS:
            raise FileNotFoundError(filename)
        target = Path(dest) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(CONTENTS[filename])
        return target


class SingleCandidateAgent:
    """只推荐假来源里的固定变体，保证三档同时订阅时只建一个下载作业。"""

    name = "test-agent"

    def recommend(self, context):
        from doomsday_seed.models import AgentSuggestion
        items = [c for c in context.candidates if c.get("eligible") and not c.get("controlled")]
        if not items:
            return []
        item = items[0]
        return [AgentSuggestion(
            repo=item["repo"], commit_rev=item["commit_rev"], variant=item["variant"],
            task=item["task"], files=item["files"], backend=item["backend"],
            estimate_basis=item["estimate_basis"],
            rationale="测试用结构化建议：许可、架构与容量预估均已过规则筛",
        )]

    def tune(self, context):
        return {}


@pytest.fixture
def service_factory(tmp_path):
    from doomsday_seed.app import Service
    from doomsday_seed.models import Settings

    created = []

    def make(profiles=("speed", "balanced", "accuracy"), agent_enabled=True,
             auto_update=True, register_runner=True):
        data = tmp_path / f"data-{len(created)}"
        api, fetch = FakeHfApi(), FakeFetch()
        svc = Service(data_dir=data, hf_api=api, fetch=fetch,
                      agent_client=SingleCandidateAgent(),
                      commit_resolver=lambda repo, ref: FIXED_SHA)
        agent = SingleCandidateAgent()
        svc.gateway.client = agent
        svc.gateway.enabled = agent_enabled
        svc.db.put_settings(Settings(
            subscriptions={"chat": list(profiles)},
            auto_update=auto_update, schedule_enabled=False,
            agent_enabled=agent_enabled, trusted_publishers=["TestOrg"],
        ))
        if register_runner:
            svc.registry.register("cpu", [sys.executable, str(MOCK_RUNNER)], "b-test")
        created.append(svc)
        return SimpleNamespace(service=svc, api=api, fetch=fetch, data=data,
                               agent=agent, paths=svc.paths)

    yield make
    for svc in created:
        try:
            svc.shutdown()
        except Exception:
            pass


@pytest.fixture(scope="module")
def installed(tmp_path_factory):
    """跑通一次完整同步，返回已归档一颗种子的环境（每个测试模块共享，避免重复下载）。"""
    from doomsday_seed.app import Service
    from doomsday_seed.models import Settings

    data = tmp_path_factory.mktemp("installed") / "data"
    api, fetch = FakeHfApi(), FakeFetch()
    svc = Service(data_dir=data, hf_api=api, fetch=fetch,
                  agent_client=SingleCandidateAgent(),
                  commit_resolver=lambda repo, ref: FIXED_SHA)
    svc.gateway.client = SingleCandidateAgent()
    svc.gateway.enabled = True
    svc.db.put_settings(Settings(
        subscriptions={"chat": ["speed", "balanced", "accuracy"]},
        auto_update=True, schedule_enabled=False, agent_enabled=True,
        trusted_publishers=["TestOrg"]))
    svc.registry.register("cpu", [sys.executable, str(MOCK_RUNNER)], "b-test")
    report = svc.pipeline.sync(trigger="test", wait=True)
    assert report.status in ("ok", "partial"), report.message
    yield SimpleNamespace(service=svc, api=api, fetch=fetch, data=data,
                          paths=svc.paths)
    svc.shutdown()
