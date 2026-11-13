"""断网演练与导出恢复包（文档 §6、附录「离线种子与验收」）。

导出包包含应用、锁定依赖或可携带运行环境、推理二进制及所需库、全部模型附件、
样本、许可证、清单和本地恢复说明；网页字体、脚本、样式和背景图均本地打包。
模拟停用同步只用于自查，系统层切断外网后才记为真实断网演练。
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
import tempfile
import subprocess
import zipfile
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import Paths
from .db import Database
from .downloader import sha256_file

RESTORE_TEXT = """# 末日种子 · 本地恢复说明

本包用于「有网时备齐、断网后冷启动」。所有网页字体、脚本、样式与背景图均在包内，
断网后 UI 不发起任何外部资产请求，聊天 API 不调用云端，找不到可用模型时返回可读错误。

## 1. 目标机器先备齐

- 相容的操作系统（首版为 Windows 11 x64）、CPU 指令集；
  使用 CUDA 后端时还需要相容的 NVIDIA 显卡驱动。
- Python 3.11+（与打包时的 ABI 相容）。动作识别需要 Windows x64 / Python 3.13。
- 手工登记的外部运行器需要在目标机器准备并重新登记。导出结果会列出未随包迁移的后端。

## 2. 离线安装

在解包根目录创建并激活虚拟环境，再执行：

```powershell
pip install --no-index --find-links data/wheelhouse -r src/doomsday_seed/resources/requirements.lock.txt
$env:PYTHONPATH = (Resolve-Path src).Path
```

若没有 data/wheelhouse，请先在有网且 Python ABI 相同的机器上运行
`doomsday-seed export --wheelhouse` 备齐依赖后重新导出。解包后的程序直接从 src 加载。

## 3. 冷启动（先系统层断网）

1. 关闭正在运行的应用与推理子进程。
2. 为本次演练指定一个空的 HF_HOME 目录，避免误用旧缓存。
3. 用防火墙阻断外网或断开网络适配器，**保留回环地址 127.0.0.1**。
4. 从另一个目录解包并启动：`python -m doomsday_seed serve`。
5. 打开 `http://127.0.0.1:8777/`，验证完整 UI、试验台流式试用和 `/v1/chat/completions`。

推理进程启动前设 `HF_HUB_OFFLINE=1`，加载器使用本地模式；这些开关只约束 Hugging Face Hub
访问，整条链路仍须用系统层断网验证。若验收要求零外联尝试，另记录进程及子进程的网络事件。

## 4. 自检

- `python -m doomsday_seed verify`：对账清单，按 SHA-256 校验每个种子包。
- 删除一个必需文件再测一次，预期返回可定位的缺件错误（指明具体文件与源路径）。
- 用 `SHA256SUMS.txt` 校验复制到第二块介质的包；离线使用不需要平台登录。

## 5. 已知限制

- 网页与本地客户端只监听回环地址；首次启动会生成新的 data/api-key。外部 Agent 密钥需另行配置。
- 单次只加载一个模型；后台评测在交互时排队。
- 每项能力的评测口径、硬件门槛与试用方式见各包内 USAGE.md。
"""


class RecoveryService:
    def __init__(self, db: Database, paths: Paths, library_lock):
        self.db = db
        self.paths = paths
        self.library_lock = library_lock

    # ---------- 断网演练 ----------
    @staticmethod
    def drill_steps() -> list[dict[str, str]]:
        return [
            {"step": "1", "title": "有网时备齐",
             "detail": "导出恢复包、准备 wheelhouse、锁定运行器构建与全部模型附件；"
                       "确认网页资源均已本地打包。"},
            {"step": "2", "title": "关进程与空缓存",
             "detail": "关闭应用与推理子进程；清空用户模型缓存，从另一目录解包。"},
            {"step": "3", "title": "系统层断网",
             "detail": "用防火墙阻断外网或断开网络适配器，保留 127.0.0.1 回环连接；"
                       "界面上的「模拟停用同步」只用于自查，不能记为真实演练。"},
            {"step": "4", "title": "冷启动",
             "detail": "python -m doomsday_seed serve；推理进程设 HF_HUB_OFFLINE=1 且只用本地路径。"},
            {"step": "5", "title": "验证完整链路",
             "detail": "打开完整 UI、在试验台发起流式试用、用本地 API 调用 "
                       "/v1/chat/completions；确认无外部资产请求、无云端回退。"},
            {"step": "6", "title": "记录与第二台机器",
             "detail": "在界面点「已按步骤完成系统层断网」记录演练；在相容的干净机器或虚拟机上再做一次恢复演练。"},
        ]

    def status(self) -> dict[str, Any]:
        settings = self.db.get_settings()
        return {
            "steps": self.drill_steps(),
            "simulated_offline": settings.simulated_offline,
            "real_drill_recorded": self.db.get_meta("real_drill") == "1",
            "real_drill_at": self.db.get_meta("real_drill_at") or None,
            "text": ("外网不可达。已验证的种子可继续使用。"
                     if settings.simulated_offline else "外网可用，可先备齐恢复包。"),
            "notice": "模拟停用同步只用于自查；系统层切断外网后才记为真实断网演练。",
        }

    def set_simulated(self, enabled: bool) -> dict[str, Any]:
        settings = self.db.get_settings()
        settings.simulated_offline = bool(enabled)
        self.db.put_settings(settings)
        return self.status()

    def record_real_drill(self) -> dict[str, Any]:
        from .models import utc_now
        self.db.set_meta("real_drill", "1")
        self.db.set_meta("real_drill_at", utc_now())
        return self.status()

    # ---------- 恢复包 ----------
    def build_wheelhouse(self) -> dict[str, Any]:
        """有网时备齐：pip download 锁定依赖到 data/wheelhouse。"""
        resources = self.paths.package / "resources"
        lock = resources / "requirements.lock.txt"
        target = self.paths.data / "wheelhouse"
        target.mkdir(parents=True, exist_ok=True)
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "download", "--only-binary=:all:", "-r", str(lock), "-d", str(target)],
                capture_output=True, text=True, timeout=900)
            return {"ok": result.returncode == 0, "path": str(target),
                    "detail": (result.stdout + result.stderr)[-1000:]}
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"ok": False, "reason": str(exc)}

    def export(self, out_path: str | None = None, include_seeds: bool = True) -> dict[str, Any]:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        target = Path(out_path).resolve() if out_path else (
            self.paths.exports / f"doomsday-seed-recovery-{stamp}.zip")
        target.parent.mkdir(parents=True, exist_ok=True)
        with self.library_lock(wait=True), tempfile.TemporaryDirectory(dir=target.parent) as temp:
            return self._export(Path(temp), target, include_seeds)

    def _export(self, temp: Path, target: Path, include_seeds: bool) -> dict[str, Any]:
        from .runtime_registry import RuntimeRegistry

        warnings: list[str] = []
        entries: list[tuple[Path, str]] = []

        def add_dir(base: Path, arc_prefix: str, *, seed_files=False):
            for path in sorted(base.rglob("*")):
                if not path.is_file() or path == target or path.is_relative_to(temp):
                    continue
                rel = path.relative_to(base)
                if seed_files:
                    if rel.parts[0].startswith((".previous-", ".interrupted-")):
                        continue
                elif any((part.startswith(".") and part != ".ready") or part == "__pycache__"
                         or part.endswith(".lock") for part in rel.parts):
                    continue
                if not path.resolve().is_relative_to(base.resolve()):
                    warnings.append(f"跳过指向目录外的链接：{path}")
                    continue
                entries.append((path, f"{arc_prefix}/{rel.as_posix()}"))

        add_dir(self.paths.package, "src/doomsday_seed")
        add_dir(self.paths.root / "tests" / "fixtures", "tests/fixtures")
        add_dir(self.paths.runtimes, "data/runtimes")
        wheelhouse = self.paths.data / "wheelhouse"
        if wheelhouse.exists():
            add_dir(wheelhouse, "data/wheelhouse")
        else:
            warnings.append("未提供 wheelhouse：有网时执行 export --wheelhouse")
        if include_seeds:
            add_dir(self.paths.seeds, "data/seeds", seed_files=True)
            if not any(arc.startswith("data/seeds/") for _, arc in entries):
                warnings.append("尚无已归档种子：恢复包只能验证界面与 API 错误路径")

        snapshot = temp / self.paths.db.name
        with closing(sqlite3.connect(self.paths.db)) as source, closing(sqlite3.connect(snapshot)) as backup, backup:
            source.backup(backup)
            backup.execute("DELETE FROM settings WHERE key IN ('download_root', 'agent_base_url', 'agent_model')")
            backup.execute("UPDATE settings SET value='false' WHERE key IN ('auto_update', 'schedule_enabled')")
            backup.execute("INSERT OR REPLACE INTO settings VALUES ('simulated_offline','true')")
            backup.execute("DELETE FROM jobs")
            backup.execute("DELETE FROM meta")
            backup.execute("DELETE FROM api_keys")
            if not include_seeds:
                backup.execute("DELETE FROM seeds")
                backup.execute("DELETE FROM defaults")
                backup.execute("UPDATE candidates SET state='candidate' WHERE state IN ('downloaded','offline_ready')")
        entries.append((snapshot, f"data/{snapshot.name}"))

        registry = RuntimeRegistry(self.paths).load()
        bundled = {arc for _, arc in entries}
        for backend, entry in registry["backends"].items():
            command = []
            for part in entry.get("command", []):
                path = Path(part)
                if part == "@python" or (path.is_absolute() and path.resolve() == Path(sys.executable).resolve()):
                    part = "@python"
                elif path.is_absolute():
                    path = path.resolve()
                    if path.is_relative_to(self.paths.runtimes.resolve()):
                        part = "@runtime/" + path.relative_to(self.paths.runtimes.resolve()).as_posix()
                    elif path.is_relative_to(self.paths.fixtures.resolve()):
                        part = "@fixture/" + path.relative_to(self.paths.fixtures.resolve()).as_posix()
                    else:
                        command = []
                        break
                arc = (part.replace("@runtime/", "data/runtimes/", 1) if part.startswith("@runtime/")
                       else part.replace("@fixture/", "tests/fixtures/", 1))
                if part.startswith(("@runtime/", "@fixture/")) and arc not in bundled:
                    command = []
                    break
                command.append(part)
            if entry.get("command") and not command:
                warnings.append(f"{backend} 运行器未随包迁移，请在目标机器重新登记")
                entry.clear()
            if command and command[0] == "@python":
                entry.get("digests", {}).pop("0", None)
            entry.update(command=command, verified=False)
        registry_file = temp / "registry.json"
        registry_file.write_text(json.dumps(registry, ensure_ascii=False), encoding="utf-8")
        entries = [(path, arc) for path, arc in entries if arc != "data/runtimes/registry.json"]
        entries.append((registry_file, "data/runtimes/registry.json"))

        sums: list[str] = []
        pending = temp / "recovery.zip"
        with zipfile.ZipFile(pending, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for path, arc in entries:
                digest = hashlib.sha256()
                with path.open("rb") as source, zf.open(arc, "w", force_zip64=True) as dest:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        dest.write(block)
                        digest.update(block)
                sums.append(f"{digest.hexdigest()}  {arc}")
            zf.writestr("RESTORE.md", RESTORE_TEXT)
            zf.writestr("SHA256SUMS.txt", "\n".join(sums) + "\n")
        os.replace(pending, target)
        return {
            "path": str(target), "bytes": target.stat().st_size,
            "sha256": sha256_file(target), "files": len(entries),
            "warnings": warnings,
            "offline_assets": "网页资产均为本地文件；CSP 禁止外部资源",
        }

    @staticmethod
    def verify_package(package: Path) -> dict[str, Any]:
        """复制到第二块介质后用 SHA-256 校验，缺件错误可定位到具体文件。"""
        problems: list[str] = []
        with zipfile.ZipFile(package) as zf:
            names = set(zf.namelist())
            if "RESTORE.md" not in names or "SHA256SUMS.txt" not in names:
                problems.append("恢复包缺少 RESTORE.md 或 SHA256SUMS.txt")
                return {"ok": False, "problems": problems}
            lines = zf.read("SHA256SUMS.txt").decode("utf-8").splitlines()
            for line in lines:
                if not line.strip():
                    continue
                digest, arc = line.split("  ", 1)
                if arc not in names:
                    problems.append(f"缺少必需文件：{arc}")
                    continue
                with zf.open(arc) as source:
                    digest_stream = hashlib.sha256()
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        digest_stream.update(block)
                actual = digest_stream.hexdigest()
                if actual != digest:
                    problems.append(f"{arc} SHA-256 不符")
        return {"ok": not problems, "problems": problems, "checked": len(lines)}
