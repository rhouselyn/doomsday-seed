"""§6、§7 与附录：恢复包、断网说明、零外部资产与清单校验。"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path

from doomsday_seed.recovery import RecoveryService

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "src" / "doomsday_seed" / "web"


def test_export_package_contains_everything_and_verifies(installed):
    svc = installed.service
    result = svc.recovery.export(include_seeds=True)
    package = Path(result["path"])
    assert package.exists() and result["files"] > 10
    with zipfile.ZipFile(package) as zf:
        names = set(zf.namelist())
        assert "RESTORE.md" in names and "SHA256SUMS.txt" in names
        assert "src/doomsday_seed/web/index.html" in names and "src/doomsday_seed/web/assets/vault.webp" in names
        assert "src/doomsday_seed/web/assets/styles.css" in names and "src/doomsday_seed/web/assets/app.js" in names
        assert any(n.startswith("data/seeds/") and n.endswith("manifest.json") for n in names)
        assert any(n.startswith("data/seeds/") and n.endswith("USAGE.md") for n in names)
        assert any(n.startswith("src/doomsday_seed/") for n in names)
        restore = zf.read("RESTORE.md").decode("utf-8")
        assert "pip install --no-index --find-links" in restore
        assert "HF_HUB_OFFLINE" in restore and "127.0.0.1" in restore
    check = RecoveryService.verify_package(package)
    assert check["ok"] and check["checked"] == result["files"]


def test_package_checksum_detects_missing_file(installed):
    package = Path(installed.service.recovery.export(include_seeds=True)["path"])
    with zipfile.ZipFile(package) as zf:
        lines = zf.read("SHA256SUMS.txt").decode("utf-8").splitlines()
        seed_line = next(line for line in lines if "manifest.json" in line)
        digest, arc = seed_line.split("  ", 1)
    # 模拟复制到第二块介质时丢失一个必需文件。
    damaged = package.parent / "damaged.zip"
    with zipfile.ZipFile(package) as src, zipfile.ZipFile(damaged, "w") as dst:
        for info in src.infolist():
            if info.filename == arc:
                continue
            dst.writestr(info, src.read(info.filename))
    result = RecoveryService.verify_package(damaged)
    assert not result["ok"]
    assert any(arc in problem for problem in result["problems"])
    damaged.unlink(missing_ok=True)


def test_web_has_no_external_assets():
    files = [WEB / "index.html", WEB / "assets/styles.css", WEB / "assets/app.js"]
    for path in files:
        assert path.exists(), path
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"https?://", text), f"{path.name} 含外部地址"
        assert not re.search(r"//cdn\.|fonts\.googleapis|unpkg|jsdelivr", text)
    css = (WEB / "assets/styles.css").read_text(encoding="utf-8")
    assert "prefers-reduced-motion" in css and ":focus-visible" in css
    html = (WEB / "index.html").read_text(encoding="utf-8")
    assert 'src="/assets/vault.webp"' in html and (WEB / "assets/vault.webp").is_file()
    assert "断网演练" not in html
    assert 'id="view-settings"' in html
    assert 'id="view-discover"' not in html and 'id="view-agent"' not in html
    assert 'id="view-vault"' in html and 'id="view-lab"' in html
    js = (WEB / "assets/app.js").read_text(encoding="utf-8")
    assert "X-DS-Session" in js and "data:" in js and "stream: true" in js


def test_drill_steps_and_simulated_offline_are_separate(service_factory):
    env = service_factory()
    recovery = env.service.recovery
    assert len(recovery.drill_steps()) == 6
    assert recovery.status()["real_drill_recorded"] is False
    recovery.set_simulated(True)
    assert recovery.status()["simulated_offline"] is True
    assert "外网不可达" in recovery.status()["text"]
    assert recovery.status()["real_drill_recorded"] is False
    recovery.record_real_drill()
    assert recovery.status()["real_drill_recorded"] is True
    assert recovery.status()["real_drill_at"]


def test_csp_forbids_external_resources(installed):
    from fastapi.testclient import TestClient
    from doomsday_seed.app import create_app
    client = TestClient(create_app(installed.service), base_url="http://127.0.0.1")
    csp = client.get("/").headers["Content-Security-Policy"]
    assert "default-src 'self'" in csp
    assert "img-src 'self' data:" in csp
    assert "connect-src 'self'" in csp
