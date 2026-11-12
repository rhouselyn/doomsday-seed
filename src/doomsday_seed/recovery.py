"""断网演练与导出恢复包（文档 §6、附录「离线种子与验收」）。

导出包包含应用、锁定依赖或可携带运行环境、推理二进制及所需库、全部模型附件、
样本、许可证、清单和本地恢复说明；网页字体、脚本、样式和背景图均本地打包。
模拟停用同步只用于自查，系统层切断外网后才记为真实断网演练。
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import LOOPBACK_HOST, Paths
from .db import Database

RESTORE_TEXT = """# 末日种子 · 本地恢复说明

本包用于「有网时备齐、断网后冷启动」。所有网页字体、脚本、样式与背景图均在包内，
断网后 UI 不发起任何外部资产请求，聊天 API 不调用云端，找不到可用模型时返回可读错误。

## 1. 目标机器先备齐

- 相容的操作系统（首版为 Windows 11 x64）、CPU 指令集；
  使用 CUDA 后端时还需要相容的 NVIDIA 显卡驱动。
- Python 3.11+（与打包时的 ABI 相容）。锁定版本见 requirements.lock.txt。
- 恢复说明里写明预先要备齐的安装件：如需系统解码器或原生库，见 runtimes/ 目录。

## 2. 离线安装

```bash
pip install --no-index --find-links wheelhouse -r requirements.lock.txt
pip install --no-index --find-links wheelhouse -e .
```

若未提供 wheelhouse，请事先在有网机器执行：

```bash
pip download -r requirements.lock.txt -d wheelhouse
```

## 3. 冷启动（先系统层断网）

1. 关闭正在运行的应用与推理子进程。
2. 清空用户模型缓存（例如 HF_HOME 下的 hub 缓存），避免误用缓存。
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

- 网页与本地客户端只监听回环地址；其他客户端使用包内 api-key 对应的本地密钥。
- 单次只加载一个模型；后台评测在交互时排队。
- 每项能力的评测口径、硬件门槛与试用方式见各包内 USAGE.md。
"""


class RecoveryService:
    def __init__(self, db: Database, paths: Paths):
        self.db = db
        self.paths = paths

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
        lock = self.paths.root / "requirements.lock.txt"
        if not lock.exists():
            return {"ok": False, "reason": "缺少 requirements.lock.txt"}
        target = self.paths.data / "wheelhouse"
        target.mkdir(parents=True, exist_ok=True)
        import sys
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "download", "-r", str(lock), "-d", str(target)],
                capture_output=True, text=True, timeout=900)
            return {"ok": result.returncode == 0, "path": str(target),
                    "detail": (result.stdout + result.stderr)[-1000:]}
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"ok": False, "reason": str(exc)}

    def export(self, out_path: str | None = None, include_seeds: bool = True) -> dict[str, Any]:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        target = Path(out_path) if out_path else (
            self.paths.exports / f"doomsday-seed-recovery-{stamp}.zip")
        target.parent.mkdir(parents=True, exist_ok=True)
        warnings: list[str] = []
        entries: list[tuple[Path, str]] = []

        def add_dir(base: Path, arc_prefix: str, skip: set[str] | None = None):
            if not base.exists():
                warnings.append(f"目录缺失，未打包：{base}")
                return
            for path in sorted(base.rglob("*")):
                if not path.is_file():
                    continue
                rel = path.relative_to(base)
                if any(part.startswith(".") or part == "__pycache__" for part in rel.parts):
                    continue
                if skip and any(rel.match(pattern) for pattern in skip):
                    continue
                entries.append((path, f"{arc_prefix}/{rel.as_posix()}"))

        add_dir(self.paths.root / "src" / "doomsday_seed", "src/doomsday_seed")
        add_dir(self.paths.web, "web")
        add_dir(self.paths.root / "docs", "docs")
        add_dir(self.paths.root / "tests" / "fixtures", "tests/fixtures")
        add_dir(self.paths.runtimes, "runtimes")
        wheelhouse = self.paths.data / "wheelhouse"
        if wheelhouse.exists():
            add_dir(wheelhouse, "wheelhouse")
        else:
            warnings.append("未提供 wheelhouse：请先在有网时执行 pip download，或随包携带可移植运行环境")
        for name in ("pyproject.toml", "requirements.txt", "requirements.lock.txt"):
            path = self.paths.root / name
            if path.exists():
                entries.append((path, name))
        if include_seeds:
            add_dir(self.paths.seeds, "seeds")
            if not any(p[1].startswith("seeds/") for p in entries):
                warnings.append("尚无已归档种子：恢复包只能验证界面与 API 错误路径")
        api_key = self.paths.api_key
        if api_key.exists():
            entries.append((api_key, "api-key"))

        # 清单与说明（SHA256SUMS 在写完所有文件后生成并追加）。
        sums: list[str] = []
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for path, arc in entries:
                zf.write(path, arc)
                sums.append(f"{_sha256(path)}  {arc}")
            zf.writestr("RESTORE.md", RESTORE_TEXT)
            zf.writestr("SHA256SUMS.txt", "\n".join(sums) + "\n")
        return {
            "path": str(target), "bytes": target.stat().st_size,
            "sha256": _sha256(target), "files": len(entries),
            "warnings": warnings,
            "offline_assets": "web/ 内字体、脚本、样式与背景图均为本地文件；CSP 禁止外部资源",
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
                actual = hashlib.sha256(zf.read(arc)).hexdigest()
                if actual != digest:
                    problems.append(f"{arc} SHA-256 不符")
        return {"ok": not problems, "problems": problems, "checked": len(lines)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
