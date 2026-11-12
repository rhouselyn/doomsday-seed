"""§5、§8 阶段 1 关键验收：下载上限与跨档去重、归档、默认版本与恢复。"""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from doomsday_seed.archive import compute_completeness, verify_seed
from doomsday_seed.catalog import load_eval_config
from doomsday_seed.discovery import DiscoveryResult
from doomsday_seed.models import EvaluationRecord, Evidence, Job, utc_now
from tests.conftest import FIXED_SHA, REPO, VARIANT


def test_simulated_offline_blocks_new_downloads_and_discovery(service_factory):
    env = service_factory()
    svc = env.service
    svc.recovery.set_simulated(True)
    report = svc.pipeline.sync()
    assert report.status == "skipped"
    assert env.api.calls["list_models"] == 0
    with pytest.raises(ValueError, match="暂停新下载"):
        svc.pipeline.submit_download("candidate")


def test_retry_restarts_same_waiting_job(service_factory, monkeypatch):
    env = service_factory()
    svc = env.service
    result = svc.source.discover(svc.db.get_settings())
    candidate = svc.db.upsert_candidate(result.candidates[0])
    job = svc.pipeline.submit_download(candidate.id, autostart=False)
    svc.db.update_job(job.id, state="waiting_network")
    started = []
    monkeypatch.setattr(svc.pipeline, "_start_worker", lambda ids, wait: started.extend(ids))

    retry = svc.pipeline.submit_download(candidate.id, profile="speed")
    assert retry.id == job.id and retry.state == "queued"
    assert started == [job.id]
    assert retry.payload["assignments"][0]["profile"] == "speed"


def test_sota_collection_ignores_local_runtime_and_keeps_seed_unverified(service_factory, monkeypatch):
    svc = service_factory(register_runner=False).service
    item = svc.source.discover(svc.db.get_settings()).candidates[0]
    item.task = "video_gen"
    item.evidence = [Evidence(
        title="Fixture", url="https://example.test/result", model_version=item.commit_rev,
        variant=item.variant, benchmark_key="fixture/test/v1", metric="accuracy", value=0.9,
    )]
    svc.db.upsert_candidate(item)
    monkeypatch.setattr(svc, "hardware", lambda: pytest.fail("SOTA collection probed local hardware"))
    monkeypatch.setattr(svc.registry, "prepare_cpu", lambda: pytest.fail("SOTA collection prepared a runtime"))

    job = svc.pipeline.submit_download(item.id, item.task, autostart=False, mode="sota")
    assert svc.pipeline.run_job(job.id).state == "done"
    seed = svc.db.find_seed_by_candidate(item.id)
    assert seed.manifest["collection_mode"] == "sota"
    assert seed.integrity == "ok" and not seed.offline_ready
    assert svc.db.get_default(item.task, "balanced").seed_id is None
    assert svc.pipeline.submit_download(item.id, item.task, autostart=False, mode="sota").id == job.id


def test_three_profiles_download_same_variant_once(service_factory):
    env = service_factory(profiles=("speed", "balanced", "accuracy"))
    report = env.service.pipeline.sync(trigger="test", wait=True)
    assert report.status in ("ok", "partial"), report.message
    db = env.service.db
    jobs = db.list_jobs()
    assert len(jobs) == 1, "三档同时订阅时相同变体只能建一个作业"
    assert jobs[0].state == "done", jobs[0].message
    assert jobs[0].stage == "完成"
    weight_calls = [c for c in env.fetch.calls if c[1] == VARIANT]
    assert len(weight_calls) == 1, "权重只下载一次"
    assert all(c[2] == FIXED_SHA for c in env.fetch.calls), "必须使用固定 commit"


def test_archive_manifest_usage_and_evaluation(service_factory):
    env = service_factory()
    env.service.pipeline.sync(trigger="test", wait=True)
    db, paths = env.service.db, env.service.paths
    seed = db.list_seeds()[0]
    assert seed.offline_ready and seed.completeness == 1.0
    directory = paths.seeds / seed.rel_dir
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for key in ("source", "fixed_revision", "variant", "relative_paths",
                "sizes_and_hashes", "license", "runner", "runtime_config",
                "evaluation", "offline_verification", "created_at"):
        assert key in manifest or key in manifest["source"], key
    assert manifest["source"]["fixed_revision"] == FIXED_SHA
    assert manifest["source"]["repo"] == REPO
    assert manifest["runner"]["command"][0] == "llama-server"
    assert any(f["kind"] == "weight" and f["sha256"] for f in manifest["sizes_and_hashes"])
    usage = (directory / "USAGE.md").read_text(encoding="utf-8")
    assert REPO in usage and FIXED_SHA in usage
    assert "/v1/chat/completions" in usage and "HF_HUB_OFFLINE=1" in usage
    assert "已知限制" in usage
    ok, missing, mismatched = verify_seed(directory, manifest)
    assert ok and not missing and not mismatched
    records = db.list_evaluations(seed.candidate_id)
    assert len(records) == 1
    record = records[0]
    assert record.eval_version == load_eval_config()["version"]
    assert record.api_ok and record.offline_ok and record.detail["stable"]
    assert record.passed_cases == record.total_cases


def test_candidate_and_job_states_are_distinguishable(service_factory):
    env = service_factory()
    env.service.pipeline.sync(trigger="test", wait=True)
    db = env.service.db
    candidate = db.get_candidate(db.list_seeds()[0].candidate_id)
    assert candidate.state == "offline_ready"
    assert db.get_job(db.list_jobs()[0].id).state == "done"


def test_defaults_pointer_rollback_and_pinning(installed):
    db = installed.service.db
    seed_id = db.list_seeds()[0].id
    for profile in ("speed", "balanced", "accuracy"):
        pointer = db.get_default("chat", profile)
        assert pointer.seed_id == seed_id
        assert pointer.source == "auto"
    # 用户固定后自动推荐不得覆盖。
    pointer = db.get_default("chat", "balanced")
    pointer.pinned = True
    pointer.previous_seed_id = "old-seed"
    db.set_default(pointer)
    record = db.list_evaluations(installed.service.pipeline.db.list_seeds()[0].candidate_id)[0]
    changed = installed.service.pipeline._update_defaults(
        "another-seed", record, [{"task": "chat", "profile": "balanced"}])
    assert changed == []
    assert db.get_default("chat", "balanced").seed_id == seed_id
    assert db.get_default("chat", "balanced").previous_seed_id == "old-seed"


def test_failed_evaluation_never_moves_default(installed):
    db = installed.service.db
    before = db.get_default("chat", "balanced").seed_id
    failing = EvaluationRecord(
        id="failing", candidate_id="new-candidate", eval_version=load_eval_config()["version"],
        backend="cpu", runner_version="b-test", hardware_fingerprint="fp",
        quality=0.2, passed_cases=1, total_cases=8, median_latency_ms=1000,
        api_ok=True, offline_ok=False, detail={"stable": True},
    )
    changed = installed.service.pipeline._update_defaults(
        "new-seed", failing, [{"task": "chat", "profile": "balanced"}])
    assert changed == []
    assert db.get_default("chat", "balanced").seed_id == before


def test_missing_file_is_locatable_and_requeues(installed):
    from doomsday_seed.archive import reconcile
    db, paths = installed.service.db, installed.service.paths
    seed = db.list_seeds()[0]
    directory = paths.seeds / seed.rel_dir
    weight = next(f["relative_path"] for f in seed.manifest["sizes_and_hashes"]
                  if f["kind"] == "weight")
    (directory / weight).unlink()
    ok, missing, _ = verify_seed(directory, seed.manifest)
    assert not ok and weight in missing[0]
    ratio, _ = compute_completeness(directory, seed.manifest)
    assert ratio < 1.0
    report = reconcile(db, paths)
    assert any(item["seed_id"] == seed.id for item in report["incomplete"])
    assert db.get_seed(seed.id).offline_ready is False
    assert db.get_candidate(seed.candidate_id).state == "downloaded"
    assert any(j.candidate_id == seed.candidate_id for j in db.list_jobs())


def test_agent_unavailable_falls_back_to_controlled_candidates(service_factory):
    env = service_factory(agent_enabled=False)
    svc = env.service
    sample = svc.hardware()
    result = svc.preselector.build(DiscoveryResult(), sample, svc.db.get_settings(),
                                   registry_version=svc.registry.version())
    assert result.controlled is True
    assert "受控候选" in result.agent_note or "停用" in result.agent_note
    assert all(p.commit_rev == FIXED_SHA for p in result.plans)


def test_jobs_recover_after_restart(service_factory):
    env = service_factory()
    db = env.service.db
    job = Job(id="interrupted", kind="install", candidate_id="c1",
              idempotency_key="huggingface|org/repo|abc|v1", state="downloading",
              stage="下载", progress=0.4, message="中断")
    db.create_job(job)
    recovered = db.recover_jobs()
    assert [j.id for j in recovered] == ["interrupted"]
    updated = db.get_job("interrupted")
    assert updated.state == "queued" and updated.stage == "排队"
    assert "按同一版本续跑" in updated.message
