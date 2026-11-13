"""Regressions found during the repository audit."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from doomsday_seed.archive import reconcile, verify_seed
from doomsday_seed.debug_tools import DebugPermissionError
from doomsday_seed.downloader import hub_fetch
from doomsday_seed.models import DownloadPlan, FileSpec, RuntimeConfig


def test_local_runtime_ignores_proxy_and_restarts(service_factory, monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    svc = service_factory().service
    report = svc.pipeline.sync(wait=True)
    job = svc.db.get_job(report.jobs[0])
    assert job.state == "done", job.message
    messages = [{"role": "user", "content": "请回复：调试就绪"}]
    assert svc.inference.chat("chat:balanced", messages)["choices"][0]["message"]["content"] == "调试就绪"
    svc.inference.stop_server(svc.inference._server)
    assert "[DONE]" in "".join(svc.inference.chat_stream("chat:balanced", messages))


def test_hub_fetch_uses_supported_arguments(tmp_path, monkeypatch):
    import huggingface_hub

    def fetch(*, repo_id, filename, revision, local_dir):
        assert (repo_id, filename, revision) == ("org/model", "model.gguf", "a" * 40)
        return str(Path(local_dir) / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fetch)
    assert hub_fetch("org/model", "model.gguf", "a" * 40, tmp_path) == tmp_path / "model.gguf"


def test_debug_rejects_extra_startup_arguments(service_factory):
    svc = service_factory().service
    config = RuntimeConfig(extra_args=["--host", "0.0.0.0"])
    with pytest.raises(DebugPermissionError):
        svc.debug_box.validate_config(config.model_dump())


def test_corrupt_staging_file_is_refetched(service_factory):
    env = service_factory()
    candidate = env.service.source.discover(env.service.db.get_settings()).candidates[0]
    plan = DownloadPlan(candidate_id=candidate.id, repo=candidate.repo,
                        commit_rev=candidate.commit_rev, files=candidate.files)
    downloader = env.service.downloader
    folder = downloader.files_dir("retry")
    folder.mkdir(parents=True)
    weight = candidate.files[0]
    (folder / weight.local_path).write_bytes(b"x" * weight.size_bytes)
    downloader.download(plan, "retry")
    assert downloader.verify(folder, plan)["ok"]
    assert not (downloader.staging_dir("retry") / "raw" / weight.repo_path).exists()


def test_fetch_from_external_cache_preserves_source(service_factory, tmp_path):
    svc = service_factory().service
    cached = tmp_path / "shared.gguf"
    cached.write_bytes(b"weights")
    svc.downloader.fetch = lambda *args: cached
    plan = DownloadPlan(candidate_id="c", repo="org/model", commit_rev="a" * 40,
                        files=[FileSpec(repo_path="model.gguf", local_path="model.gguf", size_bytes=7)])
    folder = svc.downloader.download(plan, "job")
    assert cached.read_bytes() == (folder / "model.gguf").read_bytes() == b"weights"


def test_empty_manifest_is_not_verified(tmp_path):
    assert not verify_seed(tmp_path, {})[0]


@pytest.mark.parametrize("relative", ["../outside.gguf", "C:/outside.gguf", "model:stream", "NUL", "dir/../model"])
def test_archive_paths_stay_inside_package(tmp_path, relative):
    manifest = {"sizes_and_hashes": [{"relative_path": relative, "sha256": "a" * 64}]}
    assert not verify_seed(tmp_path, manifest)[0]


@pytest.mark.parametrize("drop_job", [False, True])
def test_reconcile_repairs_completed_or_missing_job(service_factory, drop_job):
    env = service_factory()
    svc = env.service
    item = svc.source.discover(svc.db.get_settings()).candidates[0]
    svc.db.upsert_candidate(item)
    job = svc.pipeline.submit_download(item.id, mode="requested", autostart=False)
    assert svc.pipeline.run_job(job.id).state == "done"
    seed = svc.db.get_seed(item.id)
    (svc.paths.seeds / seed.rel_dir / item.files[0].local_path).unlink()
    if drop_job:
        svc.db.execute("DELETE FROM jobs WHERE id=?", (job.id,))
    report = reconcile(svc.db, svc.paths)
    assert len(report["requeued"]) == 1
    recovered = svc.db.get_job(report["requeued"][0])
    assert recovered.state == "queued" and recovered.payload["plan"]
    assert svc.pipeline.run_job(recovered.id).state == "done"
    assert verify_seed(svc.paths.seeds / seed.rel_dir, svc.db.get_seed(seed.id).manifest)[0]
    assert not svc.db.get_seed(seed.id).offline_ready


def test_download_keeps_upstream_usage_separate(service_factory, monkeypatch):
    env = service_factory()
    contents = env.fetch.__call__.__globals__["CONTENTS"]
    monkeypatch.setitem(contents, "USAGE.md", b"Upstream instructions")
    item = env.service.source.discover(env.service.db.get_settings()).candidates[0]
    env.service.db.upsert_candidate(item)
    job = env.service.pipeline.submit_download(item.id, mode="requested", autostart=False)
    assert env.service.pipeline.run_job(job.id).state == "done"
    seed = env.service.db.get_seed(item.id)
    directory = env.paths.seeds / seed.rel_dir
    assert (directory / "upstream/USAGE.md").read_bytes() == contents["USAGE.md"]
    assert verify_seed(directory, seed.manifest)[0]


def test_cli_waits_for_sync_and_download(service_factory, monkeypatch):
    from doomsday_seed.cli import build_parser, cmd_sync, cmd_download
    svc = service_factory().service
    calls = []

    def sync(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(status="ok", jobs=[], model_dump=lambda: {})

    monkeypatch.setattr(svc.pipeline, "sync", sync)
    cmd_sync(build_parser().parse_args(["sync", "--scheduled"]), svc)
    assert calls == [{"trigger": "scheduled", "wait": True}]
    candidate = svc.source.discover(svc.db.get_settings()).candidates[0]
    svc.db.upsert_candidate(candidate)
    monkeypatch.setattr(svc.pipeline, "_start_worker", lambda *a, **k: pytest.fail("CLI started daemon worker"))
    monkeypatch.setattr(svc.pipeline, "run_job", lambda job_id: svc.db.update_job(job_id, state="done"))
    assert cmd_download(build_parser().parse_args(["download", candidate.id]), svc) == 0


def test_schedule_keeps_custom_data_directory(tmp_path):
    from xml.etree import ElementTree
    from doomsday_seed.scheduler import task_xml
    data = tmp_path / "custom data & seeds"
    xml = ElementTree.fromstring(task_xml("03:00", "python.exe", tmp_path, data_dir=data))
    args = xml.find(".//{*}Arguments").text
    assert '--data "' + str(data) + '"' in args


def test_failed_publisher_does_not_discard_other_results(service_factory, monkeypatch):
    env = service_factory()
    svc = env.service
    settings = svc.db.get_settings()
    settings.trusted_publishers = ["unavailable", "available"]
    previous = "2026-11-10T01:00:00+00:00"
    svc.db.mark_source_success("huggingface", previous)
    original = env.api.list_models

    def list_models(**kwargs):
        if kwargs["author"] == "unavailable":
            raise RuntimeError("offline")
        yield from original(**kwargs)

    monkeypatch.setattr(env.api, "list_models", list_models)
    result = svc.source.discover(settings)
    assert result.candidates and result.status == "partial"
    assert svc.db.get_source("huggingface")["cursor"] == previous


def test_json_grading_rejects_scalar_and_array():
    from doomsday_seed.catalog import grade_case
    case = {"expect": {"type": "json_format", "keys": ["ok"]}}
    for output in ('null', '1', 'true', '["ok"]', '"ok"'):
        assert not grade_case(case, output)[0]


def test_accuracy_promotion_requires_two_extra_cases():
    from doomsday_seed.catalog import promote, load_eval_config
    from tests.test_core import _record
    current = _record(passed_cases=6, quality=0.75)
    challenger = _record(passed_cases=7, quality=0.875)
    assert not promote(current, challenger, "accuracy", load_eval_config())[0]


def test_cuda_estimate_and_cpu_fallback_match_runtime(service_factory):
    from doomsday_seed.models import HardwareSample
    env = service_factory()
    info = env.api.model_info("org/model")
    info.siblings[0].size = 2 * 1024**3
    item = env.service.source._to_candidates(info)[0]
    assert item.estimate.backend == "cuda" and item.estimate.est_vram_mib > 2000
    sample = HardwareSample(ram_total_mib=8192, ram_available_mib=4096,
                            vram_total_mib=4096, vram_available_mib=1024,
                            backend_available={"cpu": True, "cuda": True})
    assert not env.service.preselector.rule_filter(item, sample, env.service.db.get_settings())[0]
    sample.backend_available["cuda"] = False
    sample.ram_available_mib = 3072
    assert not env.service.preselector.rule_filter(item, sample, env.service.db.get_settings())[0]


def test_projection_is_not_selected_as_a_weight(service_factory):
    env = service_factory()
    info = env.api.model_info("org/model")
    info.siblings.insert(0, SimpleNamespace(rfilename="mmproj-model-f16.gguf", size=1, lfs={}))
    candidates = env.service.source._to_candidates(info)
    assert len(candidates) == 1
    assert not candidates[0].variant.startswith("mmproj")
    assert any(f.kind == "projection" for f in candidates[0].files)


def test_unspecified_custom_license_is_not_automatically_accepted(service_factory):
    svc = service_factory().service
    candidate = svc.source.discover(svc.db.get_settings()).candidates[0]
    candidate.license = "other"
    svc.db.upsert_candidate(candidate)
    with pytest.raises(ValueError, match="许可"):
        svc.pipeline.submit_download(candidate.id, autostart=False)


def test_revalidating_existing_files_needs_no_second_copy(service_factory, monkeypatch):
    svc = service_factory().service
    plan = DownloadPlan(candidate_id="c", repo="org/model", commit_rev="a" * 40,
                        files=[FileSpec(repo_path="model.gguf", local_path="model.gguf", size_bytes=1000)])
    monkeypatch.setattr("doomsday_seed.downloader.shutil.disk_usage", lambda _: SimpleNamespace(free=100))
    assert not svc.downloader.preflight(plan, "job").ok
    assert svc.downloader.preflight(plan, "job", reuse=True).ok


def test_cli_settings_moves_library_and_default_validates_task(service_factory, tmp_path):
    from doomsday_seed.cli import build_parser, cmd_settings, cmd_default
    from doomsday_seed.models import Seed
    svc = service_factory().service
    (svc.paths.seeds / "sentinel").write_text("preserve", encoding="utf-8")
    target = tmp_path / "new-library"
    args = build_parser().parse_args(["settings", "--set", f"download_root={target}"])
    assert cmd_settings(args, svc) == 0
    assert (target / "sentinel").read_text(encoding="utf-8") == "preserve"
    svc.db.put_seed(Seed(id="hand", candidate_id="hand", rel_dir="hand", offline_ready=True,
                         integrity="ok", manifest={"task": "hand"}))
    args = build_parser().parse_args(["default", "chat", "balanced", "hand"])
    assert cmd_default(args, svc) == 1
    assert svc.db.get_default("chat", "balanced").seed_id is None


def test_cli_ndjson_is_one_object_per_line(service_factory, capsys):
    from doomsday_seed.cli import build_parser, cmd_list
    svc = service_factory().service
    cmd_list(build_parser().parse_args(["list", "--json-ndjson"]), svc)
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == len(svc.db.list_candidates())
    assert all(isinstance(json.loads(line), dict) for line in lines)


def test_incomplete_models_and_aliases_are_not_advertised(service_factory):
    from doomsday_seed.models import Seed, DefaultPointer
    svc = service_factory().service
    svc.db.put_seed(Seed(id="bad", candidate_id="bad", offline_ready=True, integrity="incomplete"))
    svc.db.set_default(DefaultPointer(task="chat", profile="balanced", seed_id="bad"))
    assert svc.inference.models() == []
    from doomsday_seed.runner import RuntimeUnavailable
    with pytest.raises(RuntimeUnavailable):
        svc.inference._load("chat:balanced")


def test_status_counts_jobs_older_than_the_display_limit(service_factory):
    from doomsday_seed.api import status
    from doomsday_seed.models import Job
    svc = service_factory().service
    svc.db.create_job(Job(id="active", state="downloading", created_at="2020-01-01"))
    for index in range(25):
        svc.db.create_job(Job(id=str(index), state="done"))
    assert status(svc)["counts"]["active_jobs"] == 1


def test_invalid_schedule_is_rejected_before_settings_change(service_factory):
    from fastapi.testclient import TestClient
    from doomsday_seed.app import create_app
    svc = service_factory().service
    client = TestClient(create_app(svc), base_url="http://127.0.0.1")
    before = svc.db.get_settings()
    result = client.put("/api/settings", json={"schedule_time": "25:99"},
                        headers={"X-DS-Session": svc.security.session_token})
    assert result.status_code == 422
    assert svc.db.get_settings() == before


def test_failed_jobs_do_not_record_successful_sync(service_factory, monkeypatch):
    from doomsday_seed.jobs import JobResult
    from doomsday_seed.models import SyncReport
    svc = service_factory().service
    report = SyncReport(id="failed-sync", status="ok", jobs=["job"])
    svc.db.start_sync(report)
    monkeypatch.setattr(svc.pipeline, "run_job", lambda _: JobResult("job", "waiting_network", "offline"))
    svc.pipeline._run_jobs(report.jobs, report)
    assert report.status == "failed"
    assert svc.db.get_meta("last_sync_success") == ""
    assert svc.db.list_sync_runs()[0]["status"] == "failed"


def test_auto_default_skips_higher_scoring_failed_evaluation(service_factory):
    from doomsday_seed.api import DefaultPayload, put_default
    from doomsday_seed.models import Seed
    from tests.test_core import _record
    svc = service_factory().service
    fingerprint = svc.hardware().fingerprint()
    for name, score, api_ok in (("failed", 0.99, False), ("usable", 0.8, True)):
        svc.db.put_seed(Seed(id=name, candidate_id=name, offline_ready=True, integrity="ok",
                             manifest={"task": "chat"}))
        svc.db.add_evaluation(_record(id=name, candidate_id=name, score_balanced=score, api_ok=api_ok,
                                      hardware_fingerprint=fingerprint))
    result = put_default("chat", "balanced", DefaultPayload(auto=True), svc)
    assert result["default"]["seed_id"] == "usable"


def test_partial_discovery_without_jobs_does_not_record_sync_success(service_factory, monkeypatch):
    from doomsday_seed.discovery import DiscoveryResult
    from doomsday_seed.preselect import ShortlistResult
    svc = service_factory(profiles=()).service
    monkeypatch.setattr(svc.source, "discover", lambda _: DiscoveryResult(
        errors=[{"error": "offline"}], status="partial"))
    monkeypatch.setattr(svc.preselector, "build", lambda *a, **kw: ShortlistResult())
    report = svc.pipeline.sync(wait=True)
    assert report.status == "partial" and not report.jobs
    assert svc.db.get_meta("last_sync_success") == ""


def test_collected_chat_can_be_validated_without_redownload(service_factory):
    env = service_factory()
    svc = env.service
    candidate = svc.source.discover(svc.db.get_settings()).candidates[0]
    svc.db.upsert_candidate(candidate)
    job = svc.pipeline.submit_download(candidate.id, mode="requested", autostart=False)
    assert svc.pipeline.run_job(job.id).state == "done"
    before = len(env.fetch.calls)
    upgraded = svc.pipeline.submit_download(candidate.id, mode="local", autostart=False)
    result = svc.pipeline.run_job(upgraded.id)
    assert result.state == "done", result.message
    assert svc.db.get_seed(candidate.id).offline_ready
    assert len(env.fetch.calls) == before
