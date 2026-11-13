from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from doomsday_seed.agent_interaction import _command, _run_tool
from doomsday_seed.app import create_app
from doomsday_seed.credentials import CredentialStore
from doomsday_seed.local_import import enqueue


@pytest.mark.skipif(os.name != "nt", reason="Windows DPAPI")
def test_key_is_encrypted_and_never_returned(service_factory):
    svc = service_factory().service
    client = TestClient(create_app(svc), base_url="http://127.0.0.1")
    token = client.get("/api/session").json()["session_token"]
    key = "sk-test-for-local-test-123456789"
    response = client.put("/api/agent/config", headers={"X-DS-Session": token}, json={"api_key": key})
    assert response.status_code == 200 and response.json()["configured"]
    assert key not in response.text and key not in client.get("/api/agent/config").text
    assert key.encode() not in svc.credentials.path.read_bytes()
    assert CredentialStore(svc.paths.data).get() == key


def test_agent_actions_require_explicit_matching_request(service_factory):
    svc = service_factory().service
    assert _command(svc, "请解释 org/repo 的下载方法，先只讲步骤。") is None
    assert _command(svc, "请归档 C:/models/a.gguf") is None
    with pytest.raises(ValueError, match="不支持"):
        _run_tool(svc, "archive_local", {"path": "C:/models", "task": "chat"})
    assert not svc.db.list_jobs()


def test_local_archive_copies_and_preserves_source(service_factory, tmp_path, monkeypatch):
    svc = service_factory().service
    source = tmp_path / "existing model"
    source.mkdir()
    (source / "model.gguf").write_bytes(b"model bytes")
    (source / "USAGE.md").write_text("original notes", encoding="utf-8")
    monkeypatch.setattr(svc.pipeline, "_start_worker", lambda *args, **kwargs: None)
    job = enqueue(svc, str(source), "chat")
    result = svc.pipeline.run_job(job.id)
    assert result.state == "done", result.message
    seed = svc.db.get_seed(result.seed_id)
    assert seed.integrity == "ok" and not seed.offline_ready
    assert (source / "model.gguf").read_bytes() == b"model bytes"
    assert (svc.paths.seeds / seed.rel_dir / "payload" / "USAGE.md").read_text(encoding="utf-8") == "original notes"
    second = enqueue(svc, str(source), "chat")
    assert svc.pipeline.run_job(second.id).state == "done"
    with pytest.raises(ValueError):
        enqueue(svc, str(svc.paths.data), "chat")


def test_discovery_records_metadata_without_downloads(service_factory, monkeypatch):
    from doomsday_seed import researcher
    env = service_factory()
    svc = env.service
    original = env.api.model_info
    monkeypatch.setattr(env.api, "model_info", lambda repo, **kwargs: original(repo, files_metadata=True))
    monkeypatch.setattr(researcher, "RESEARCH_REPOS", {"chat": "TestOrg/offline-tiny-GGUF"})
    before = (len(svc.db.list_jobs()), len(svc.db.list_seeds()))
    report = researcher.discover(svc)
    assert report["downloaded"] is False and report["items"][0]["revision"]
    assert researcher.latest(svc) == report
    assert not env.fetch.calls and before == (len(svc.db.list_jobs()), len(svc.db.list_seeds()))


def test_agent_chat_calls_read_only_tools_without_downloading(service_factory, monkeypatch):
    from doomsday_seed.agent import DeepSeekClient
    from doomsday_seed.agent_interaction import chat
    env = service_factory()
    sent = []
    def complete(self, messages, *, tools=None, json_mode=False):
        sent.append((messages[:], tools))
        if len(sent) == 1:
            return {"tool_calls": [{"id": "call1", "type": "function", "function": {
                "name": "hardware_summary", "arguments": "{}"}}]}
        return {"content": "已读取硬件；候选尚未下载。"}
    monkeypatch.setattr(DeepSeekClient, "complete", complete)
    response = chat(env.service, "先查看本机硬件，不要下载")
    assert response["actions"][0]["tool"] == "hardware_summary"
    assert "下载" in response["answer"]
    assert not any(tool["function"]["name"] == "download_model" for tool in sent[0][1])
    assert not env.fetch.calls and not env.service.db.list_jobs()


def test_rule_agent_assigns_multiple_profiles():
    from doomsday_seed.agent import AgentContext, LocalRuleAgent
    candidates = [{"id": str(i), "repo": "test/model", "commit_rev": "a" * 40,
                   "variant": str(i), "task": "chat", "eligible": True,
                   "est_ram_mib": i * 100} for i in (1, 2, 3)]
    context = AgentContext(candidates=candidates, subscriptions=[
        ["chat", "speed"], ["chat", "balanced"], ["chat", "accuracy"]], max_suggestions=3)
    result = LocalRuleAgent().recommend(context)
    assert [(item.profile, item.variant) for item in result] == [
        ("speed", "1"), ("balanced", "2"), ("accuracy", "3")]
