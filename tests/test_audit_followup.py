from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from doomsday_seed.api import save_settings
from doomsday_seed.models import Job


def test_second_process_does_not_recover_owned_job(service_factory):
    svc = service_factory().service
    svc.db.create_job(Job(id="running", state="downloading"))
    script = """
import sys
from doomsday_seed.app import Service
svc = Service(sys.argv[1])
try:
    assert svc.pipeline.recover() == []
    assert svc.db.get_job("running").state == "downloading"
finally:
    svc.shutdown()
"""
    with svc.pipeline.library_lock():
        subprocess.run([sys.executable, "-c", script, str(svc.paths.data)], check=True, timeout=30,
                       env={**os.environ, "PYTHONPATH": str(svc.paths.package.parent)})
    assert svc.pipeline.recover() == ["running"]


def test_migration_rolls_back_when_settings_commit_fails(service_factory, tmp_path, monkeypatch):
    svc = service_factory().service
    old = svc.paths.seeds
    (old / "model").write_bytes(b"keep this")
    settings = svc.db.get_settings()
    target = tmp_path / "other-library"
    settings.download_root = str(target)

    def fail(_, **kwargs):
        raise OSError("database write failed")

    monkeypatch.setattr(svc.db, "put_settings", fail)
    with pytest.raises(OSError, match="database write failed"):
        save_settings(svc, settings)
    assert (old / "model").read_bytes() == b"keep this"
    assert not target.exists()
    assert svc.paths.seeds == old and not svc.db.get_settings().download_root


def test_replacing_archive_rolls_back_when_database_commit_fails(service_factory):
    svc = service_factory().service
    old = svc.paths.seeds / "seed"
    new = svc.downloader.files_dir("job")
    for directory, value in ((old, b"original"), (new, b"replacement")):
        directory.mkdir(parents=True)
        (directory / "model").write_bytes(value)
    with pytest.raises(OSError):
        with svc.downloader.promote("job", "seed") as promoted:
            assert (promoted / "model").read_bytes() == b"replacement"
            raise OSError("commit failed")
    assert (old / "model").read_bytes() == b"original"
    assert (new / "model").read_bytes() == b"replacement"


def test_recovery_cold_start_from_extracted_source(installed, tmp_path):
    svc = installed.service
    marker = svc.paths.runtimes / "mediapipe-test" / ".ready"
    marker.parent.mkdir(exist_ok=True)
    marker.touch()
    archive = Path(svc.recovery.export()["path"])
    restored = tmp_path / "restored"
    with zipfile.ZipFile(archive) as package:
        assert "data/runtimes/mediapipe-test/.ready" in package.namelist()
        assert "data/api-key" not in package.namelist()
        package.extractall(restored)
    script = """
from fastapi.testclient import TestClient
from doomsday_seed.app import Service, create_app
svc = Service()
assert svc.db.get_settings().simulated_offline
assert svc.registry.command("cpu")
with TestClient(create_app(svc), base_url="http://127.0.0.1") as client:
    assert client.get("/").status_code == 200
    assert client.get("/assets/vault.webp").status_code == 200
    result = client.post("/v1/chat/completions", headers={"Authorization": f"Bearer {svc.security.api_key}"},
                         json={"model": "chat:balanced", "messages": [{"role": "user", "content": "Reply OK"}]})
    assert result.status_code == 200, result.text
"""
    env = {**os.environ, "PYTHONPATH": str(restored / "src"), "DOOMSDAY_DATA": str(restored / "data")}
    subprocess.run([sys.executable, "-c", script], cwd=restored, env=env, check=True, timeout=45)


def test_changed_runner_file_cannot_start(service_factory, tmp_path):
    svc = service_factory().service
    script = tmp_path / "runner.py"
    script.write_text("print('original')", encoding="utf-8")
    svc.registry.register("cpu", [sys.executable, str(script)], "test")
    assert svc.registry.command("cpu")
    script.write_text("print('replacement')", encoding="utf-8")
    assert not svc.registry.command("cpu")
    assert not svc.registry.probe()[0]["cpu"]


def test_shards_form_one_candidate_and_incomplete_group_is_ignored(service_factory):
    env = service_factory()
    info = env.api.model_info("fixture")
    info.siblings = [SimpleNamespace(rfilename=name, size=512, lfs={}) for name in (
        "x-Q4_K_M-00002-of-00002.gguf", "x-Q4_K_M-00001-of-00002.gguf",
        "incomplete-00001-of-00003.gguf")]
    [candidate] = env.service.source._to_candidates(info)
    assert candidate.variant == "x-Q4_K_M-00001-of-00002.gguf"
    assert [f.local_path for f in candidate.files] == [
        "x-Q4_K_M-00001-of-00002.gguf", "x-Q4_K_M-00002-of-00002.gguf"]
    assert candidate.size_bytes == 1024


def test_first_discovery_reads_all_pages_and_commits_cursor_with_catalog(service_factory, monkeypatch):
    env = service_factory()
    svc = env.service
    template = env.api.model_info("fixture")
    repos = [f"TestOrg/model-{i}" for i in range(25)]
    monkeypatch.setattr(env.api, "list_models", lambda **kw: (
        SimpleNamespace(id=repo, last_modified=None) for repo in repos[:kw.get("limit")]))
    monkeypatch.setattr(env.api, "model_info", lambda repo, **kw:
                        SimpleNamespace(**(vars(template) | {"id": repo})))
    settings = svc.db.get_settings()
    result = svc.source.discover(settings)
    assert {candidate.repo for candidate in result.candidates} == set(repos)
    assert not svc.db.get_source("huggingface").get("cursor")
    svc.preselector.store_discovery(result, settings)
    assert svc.db.get_source("huggingface")["cursor"] == result.cursors["huggingface"]


def test_main_candidate_pins_metadata_before_queueing(service_factory):
    env = service_factory()
    svc = env.service
    candidate = svc.source.discover(svc.db.get_settings()).candidates[0]
    placeholder = candidate.model_copy(deep=True)
    placeholder.commit_rev = "main"
    placeholder.id = placeholder.make_id(placeholder.source, placeholder.repo, "main", placeholder.variant)
    placeholder.size_bytes = 0
    for file in placeholder.files:
        file.size_bytes = 0
    svc.db.upsert_candidate(placeholder)
    job = svc.pipeline.submit_download(placeholder.id, autostart=False)
    assert job.candidate_id == candidate.id
    assert job.idempotency_key == candidate.idempotency_key
    assert job.payload["plan"]["files"][0]["upstream_sha256"]
    assert job.payload["plan"]["total_bytes"] == candidate.size_bytes


def test_retest_clears_broken_default_and_records_integrity(service_factory):
    svc = service_factory().service
    assert svc.pipeline.sync(wait=True).status == "ok"
    seed = svc.db.list_seeds()[0]
    retest = svc.pipeline.retest(seed.id)
    assert retest["ok"]
    assert svc.db.get_seed(seed.id).manifest["evaluation"]["id"] == retest["evaluation"]["id"]
    directory = svc.paths.seeds / seed.rel_dir
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    (directory / manifest["sizes_and_hashes"][0]["relative_path"]).unlink()
    result = svc.pipeline.retest(seed.id)
    assert not result["ok"] and result["errors"]
    assert not svc.db.get_seed(seed.id).offline_ready
    assert svc.db.get_default("chat", "balanced").seed_id is None
