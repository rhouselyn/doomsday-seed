"""§6 接口验收：状态、目录、设置、默认版本、流式聊天与本地安全。"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from doomsday_seed.app import create_app


@pytest.fixture
def client(installed):
    app = create_app(installed.service)
    return TestClient(app, base_url="http://127.0.0.1")


@pytest.fixture
def token(client):
    return client.get("/api/session").json()["session_token"]


def test_local_pages_and_status(client):
    index = client.get("/")
    assert index.status_code == 200 and "末日种子" in index.text
    assert index.headers["Content-Security-Policy"].startswith("default-src 'self'")
    status = client.get("/api/status").json()
    assert status["counts"]["offline_ready"] == 1
    assert "外网" in status["network"]["text"]
    health = client.get("/api/health").json()
    assert health["ok"] and health["loopback_only"]


def test_catalog_separates_public_estimate_and_local_measurement(client):
    data = client.get("/api/catalog").json()
    assert data["eval_version"] == "stage1-1.0.0"
    ready = [c for c in data["candidates"] if c["state"] == "offline_ready"]
    assert len(ready) == 1
    item = ready[0]
    assert item["state_label"] == "离线可用" and item["state_icon"]
    assert item["public_evidence"] and item["estimate"]["label"] == "预估"
    assert item["measured_label"] == "本机实测" and item["measured"][0]["eval_version"]
    assert item["seed"]["offline_ready"] and item["seed"]["completeness"] == 1.0
    assert data["tasks"][0]["id"] == "chat" and len(data["profiles"]) == 3


def test_hardware_endpoint_reports_budget(client):
    data = client.get("/api/hardware").json()
    assert data["sample"]["cpu_model"]
    assert data["budget_cpu"]["ram_reserve_mib"] > 0
    assert "fingerprint" in data


def test_seed_detail_has_usage_and_history(client):
    seed = client.get("/api/status").json()["available_now"][0]["seed_id"]
    data = client.get("/api/seeds/" + seed).json()
    assert len(data["usage_markdown"]) > 200
    assert "/v1/chat/completions" in data["usage_markdown"]
    assert data["package_files"] and data["offline_verification"]["passed"]
    assert data["history"] and data["runner"]["version"]


def test_mutating_requests_need_local_session_or_key(client, token):
    assert client.post("/api/drill/simulate", json={"enabled": True}).status_code == 403
    ok = client.post("/api/drill/simulate", json={"enabled": True},
                     headers={"X-DS-Session": token})
    assert ok.status_code == 200
    # 其他客户端可用本地 API 密钥。
    key = installed_key(client)
    assert client.post("/api/drill/simulate", json={"enabled": False},
                       headers={"Authorization": "Bearer " + key}).status_code == 200


def installed_key(client):
    from pathlib import Path
    data_dir = Path(client.app.state.service.paths.data)
    return (data_dir / "api-key").read_text(encoding="utf-8").strip()


def test_host_and_origin_are_restricted(client):
    assert client.get("/api/health", headers={"Host": "evil.example"}).status_code == 403
    assert client.get("/api/health", headers={"Origin": "https://evil.example"}).status_code == 403
    assert client.get("/api/health", headers={"Origin": "http://127.0.0.1"}).status_code == 200


def test_openai_compatible_chat_and_stream(client, token):
    headers = {"X-DS-Session": token}
    models = client.get("/v1/models").json()["data"]
    assert any(m["id"] == "chat:balanced" and m["doomsday"]["kind"] == "alias" for m in models)
    assert any(m["doomsday"]["kind"] == "version" for m in models)
    plain = client.post("/v1/chat/completions", headers=headers, json={
        "model": "chat:balanced",
        "messages": [{"role": "user", "content": "中国的首都是哪座城市？请只回答城市名。"}]})
    assert plain.status_code == 200, plain.text
    body = plain.json()
    assert body["choices"][0]["message"]["content"] == "北京"
    assert body["doomsday"]["cloud_fallback"] is False
    stream = client.post("/v1/chat/completions", headers=headers, json={
        "model": "chat:balanced", "stream": True,
        "messages": [{"role": "user", "content": "请回复：调试就绪"}]})
    assert stream.status_code == 200
    assert "data:" in stream.text and "[DONE]" in stream.text
    import json as _json
    collected = "".join(
        _json.loads(line[5:].strip())["choices"][0]["delta"].get("content", "")
        for line in stream.text.splitlines()
        if line.startswith("data:") and "[DONE]" not in line)
    assert "调试就绪" in collected


def test_no_usable_model_is_readable_and_never_cloud(client, token):
    res = client.post("/v1/chat/completions", headers={"X-DS-Session": token}, json={
        "model": "does-not-exist", "messages": [{"role": "user", "content": "你好"}]})
    assert res.status_code == 404
    assert "本机暂无可运行模型" in res.text and "不回退到云" in res.text
    stream = client.post("/v1/chat/completions", headers={"X-DS-Session": token}, json={
        "model": "does-not-exist", "stream": True,
        "messages": [{"role": "user", "content": "你好"}]})
    assert stream.headers["content-type"].startswith("text/event-stream")
    assert '"code": "no_model"' in stream.text
    assert "[DONE]" not in stream.text


def test_unified_infer_dispatches_by_task(client, token):
    headers = {"X-DS-Session": token}
    chat = client.post("/api/infer", headers=headers, json={
        "task": "chat", "model": "chat:balanced",
        "inputs": {"messages": [{"role": "user", "content": "中国的首都是哪座城市？请只回答城市名。"}]},
    })
    assert chat.status_code == 200 and chat.json()["outputs"] == {"text": "北京"}
    unsupported = client.post("/api/infer", headers=headers, json={
        "task": "face_swap", "model": "pending", "inputs": {},
    })
    assert unsupported.status_code == 501


def test_download_preferences_are_independent_by_task(client, token):
    headers = {"X-DS-Session": token}
    before = client.get("/api/preferences").json()
    assert before["ocr"]["profiles"] == ["balanced"]
    assert before["ocr"]["mode"] == "off"
    for task, payload in [
        ("pose", {"mode": "local", "profiles": ["speed"]}),
        ("video_gen", {"mode": "sota", "profiles": []}),
    ]:
        saved = client.put(f"/api/preferences/{task}", headers=headers, json=payload)
        assert saved.status_code == 200 and all(saved.json()[k] == v for k, v in payload.items())
    prefs = client.get("/api/preferences").json()
    assert prefs["pose"]["profiles"] == ["speed"]
    assert prefs["video_gen"]["mode"] == "sota"
    assert prefs["chat"] == before["chat"]
    multi = {"mode": "local", "profiles": ["speed", "balanced", "accuracy"], "sota_enabled": True}
    assert client.put("/api/preferences/ocr", headers=headers, json=multi).json() == multi
    assert client.get("/api/preferences").json()["ocr"] == multi


def test_download_root_moves_existing_seeds(service_factory, tmp_path):
    env = service_factory()
    svc = env.service
    report = svc.pipeline.sync(trigger="test", wait=True)
    assert report.status in ("ok", "partial"), report.message
    seed = svc.db.list_seeds()[0]
    old = svc.paths.seeds / seed.rel_dir
    assert old.is_dir()
    client = TestClient(create_app(svc), base_url="http://127.0.0.1")
    token = client.get("/api/session").json()["session_token"]
    current = client.get("/api/settings").json()
    payload = {k: current[k] for k in (
        "auto_update", "schedule_enabled", "schedule_time", "agent_enabled",
        "trusted_publishers", "manual_repos", "subscriptions", "simulated_offline",
        "agent_base_url", "agent_model")}
    target = tmp_path / "moved-library"
    payload["download_root"] = str(target)
    saved = client.put("/api/settings", headers={"X-DS-Session": token}, json=payload)
    assert saved.status_code == 200, saved.text
    assert (target / seed.rel_dir / "manifest.json").is_file()
    assert not old.exists()
    assert svc.paths.seeds.resolve() == target.resolve()
    assert client.get("/api/status").json()["counts"]["offline_ready"] == 1


def test_settings_roundtrip_and_effective_state(client, token):
    headers = {"X-DS-Session": token}
    current = client.get("/api/settings").json()
    payload = {k: current[k] for k in (
        "auto_update", "schedule_enabled", "schedule_time", "agent_enabled",
        "trusted_publishers", "manual_repos", "subscriptions", "simulated_offline")}
    payload["subscriptions"] = {"chat": ["speed", "balanced", "accuracy"]}
    saved = client.put("/api/settings", headers=headers, json=payload)
    assert saved.status_code == 200, saved.text
    assert saved.json()["subscriptions"]["chat"] == ["speed", "balanced", "accuracy"]
    assert "effective" in saved.json()
    assert len(client.get("/api/settings").json()["subscriptions"]["chat"]) == 3


def test_defaults_switch_and_auto_recommend(client, token):
    headers = {"X-DS-Session": token}
    seed = client.get("/api/status").json()["available_now"][0]["seed_id"]
    fixed = client.put("/api/defaults/chat/balanced", headers=headers,
                       json={"seed_id": seed, "pinned": True})
    assert fixed.status_code == 200
    assert fixed.json()["default"]["pinned"] is True
    auto = client.put("/api/defaults/chat/balanced", headers=headers, json={"auto": True})
    assert auto.status_code == 200
    assert auto.json()["default"]["pinned"] is False
    assert auto.json()["default"]["seed_id"] == seed
    bad = client.put("/api/defaults/chat/balanced", headers=headers,
                     json={"seed_id": "missing-seed"})
    assert bad.status_code == 400


def test_drill_and_unsupported_capability(client, token):
    drill = client.get("/api/drill").json()
    assert len(drill["steps"]) == 6
    assert "模拟停用同步只用于自查" in drill["notice"]
    assert client.post("/api/drill/record", headers={"X-DS-Session": token}).status_code == 200
    assert client.get("/api/drill").json()["real_drill_recorded"] is True
    ocr = client.post("/api/ocr", headers={"X-DS-Session": token})
    assert ocr.status_code == 501
    assert "阶段 2" in ocr.text


def test_upload_validation_is_enforced_locally(client, token):
    headers = {"X-DS-Session": token}
    ok = client.post("/api/uploads", headers=headers,
                     json={"filename": "notes.txt", "content": "本机文本，不外发"})
    assert ok.status_code == 200
    assert ok.json()["retained"] is False and ok.json()["local_only"] is True
    assert ok.json()["text"] == "本机文本，不外发"
    bad_type = client.post("/api/uploads", headers=headers,
                           json={"filename": "run.exe", "content": "x"})
    assert bad_type.status_code == 415
    oversized = client.post("/api/uploads", headers=headers,
                            json={"filename": "big.txt", "content": "x" * (2 * 1024 * 1024 + 1)})
    assert oversized.status_code == 413


def test_jobs_and_sync_endpoints(client, token):
    headers = {"X-DS-Session": token}
    jobs = client.get("/api/jobs").json()["items"]
    assert jobs and jobs[0]["state"] == "done"
    assert client.get("/api/jobs/" + jobs[0]["id"]).status_code == 200
    assert client.get("/api/jobs/not-there").status_code == 404
    # 已归档版本直接复用，重复同步不会新建下载作业。
    assert client.post("/api/sync?wait=true", headers=headers).status_code == 200
    assert len(client.get("/api/jobs").json()["items"]) == len(jobs)
