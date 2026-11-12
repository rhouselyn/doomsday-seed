"""运行器构建的锁定登记（文档 §2、§3、§5.1）。

「安装时锁定应用依赖、运行器构建和兼容配置，模型更新不顺带更新运行器」；
Agent 的调试工具「只调用已安装的受控运行器」，需要新运行器时作业标「待适配」。
command 允许登记为 [可执行文件] 或 [python, 脚本]（后者供 fixtures 与断网演练使用）。
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from threading import Lock
from typing import Any

from .config import Paths

CPU_BUILD = "b12207"
CPU_DIGEST = "24447c546dca02bc449c7369eba07608cf5b170a59042ef388712654356d0d4c"
_prepare_lock = Lock()

DEFAULT_REGISTRY: dict[str, Any] = {
    "locked": False,
    "name": "llama.cpp llama-server",
    "version": "unregistered",
    "build": "",
    "backends": {
        "cpu": {"command": [], "verified": False, "note": ""},
        "cuda": {"command": [], "verified": False, "note": ""},
    },
}


def _default_registry(paths: Paths) -> dict[str, Any]:
    data = json.loads(json.dumps(DEFAULT_REGISTRY))
    exe = paths.runtimes / "llama-server" / "llama-server.exe"
    cuda = paths.runtimes / "llama-server-cuda" / "llama-server.exe"
    data["backends"]["cpu"]["command"] = [str(exe)] if exe.exists() else []
    data["backends"]["cuda"]["command"] = [str(cuda)] if cuda.exists() else []
    return data


class RuntimeRegistry:
    def __init__(self, paths: Paths):
        self.paths = paths
        paths.runtimes.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, Any]:
        if self.paths.runner_registry.exists():
            try:
                data = json.loads(self.paths.runner_registry.read_text(encoding="utf-8"))
                for backend in ("cpu", "cuda"):
                    data.setdefault("backends", {}).setdefault(
                        backend, {"command": [], "verified": False, "note": ""})
                return data
            except (OSError, ValueError):
                pass
        return _default_registry(self.paths)

    def save(self, data: dict[str, Any]) -> dict[str, Any]:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.paths.runtimes, delete=False) as handle:
            pending = Path(handle.name)
            json.dump(data, handle, ensure_ascii=False, indent=2)
        try:
            pending.replace(self.paths.runner_registry)
        finally:
            pending.unlink(missing_ok=True)
        return data

    def prepare_cpu(self) -> None:
        """Install only the pinned, verified upstream Windows CPU build."""
        import httpx
        from .downloader import sha256_file

        with _prepare_lock:
            if self.command("cpu") or self.command("cuda"):
                return
            if sys.platform != "win32" or platform.machine().lower() not in ("amd64", "x86_64"):
                raise RuntimeError("当前系统的对话运行环境暂未接入")
            target = self.paths.runtimes / f"llama-{CPU_BUILD}-cpu"
            binary = target / "llama-server.exe"
            if not binary.is_file():
                archive = f"llama-{CPU_BUILD}-bin-win-cpu-x64.zip"
                url = f"https://github.com/ggml-org/llama.cpp/releases/download/{CPU_BUILD}/{archive}"
                with tempfile.TemporaryDirectory(dir=self.paths.runtimes) as temp:
                    path = Path(temp) / archive
                    with httpx.stream("GET", url, follow_redirects=True, timeout=25) as response:
                        response.raise_for_status()
                        with path.open("wb") as out:
                            for block in response.iter_bytes():
                                if out.tell() + len(block) > 200 * 1024**2:
                                    raise RuntimeError("运行器文件超过允许大小")
                                out.write(block)
                    if sha256_file(path) != CPU_DIGEST:
                        raise RuntimeError("运行器完整性校验失败")
                    unpack = Path(temp) / "verified"
                    unpack.mkdir()
                    with zipfile.ZipFile(path) as bundle:
                        for entry in bundle.infolist():
                            name = Path(entry.filename).name
                            if Path(name).suffix.lower() not in (".exe", ".dll") or ":" in name:
                                continue
                            with bundle.open(entry) as source, (unpack / name).open("xb") as out:
                                shutil.copyfileobj(source, out)
                    test = subprocess.run([str(unpack / binary.name), "--version"], env=_offline_env(),
                                          capture_output=True, timeout=12)
                    if test.returncode:
                        raise RuntimeError("对话运行环境自检失败")
                    try:
                        os.replace(unpack, target)
                    except OSError:
                        if not binary.is_file():
                            raise
            self.register("cpu", [str(binary)], CPU_BUILD)

    def command(self, backend: str) -> list[str]:
        entry = self.load()["backends"].get(backend, {})
        cmd = entry.get("command") or []
        return [str(part) for part in cmd]

    def version(self) -> str:
        return str(self.load().get("version", "unregistered"))

    def is_verified(self, backend: str) -> bool:
        return bool(self.load()["backends"].get(backend, {}).get("verified"))

    def mark_verified(self, backend: str, note: str = "") -> None:
        """§2：有「验证过的」CUDA 后端时才优先用独显；CPU 是基础回退。"""
        data = self.load()
        entry = data["backends"][backend]
        entry["verified"] = True
        entry["note"] = note
        self.save(data)

    def register(self, backend: str, command: list[str], version: str,
                 build: str = "", locked: bool = True) -> dict[str, Any]:
        data = self.load()
        data["version"] = version
        data["build"] = build or data.get("build", "")
        data["locked"] = locked
        entry = data["backends"][backend]
        if entry.get("command") != list(command):
            entry["verified"] = False  # 换了构建必须重新验证
        entry["command"] = list(command)
        entry["note"] = "已登记，等待本机冷启动验证"
        return self.save(data)

    def probe(self) -> tuple[dict[str, bool], dict[str, str]]:
        """检查后端能否真正初始化，返回 (可用性, 原因)。"""
        data = self.load()
        available: dict[str, bool] = {}
        detail: dict[str, str] = {}
        for backend in ("cpu", "cuda"):
            entry = data["backends"][backend]
            cmd = entry.get("command") or []
            if not cmd:
                available[backend] = False
                detail[backend] = "运行器未登记：请先安装锁定的 llama-server 构建"
                continue
            exe = Path(cmd[0])
            if not exe.exists() and not _on_path(cmd[0]):
                available[backend] = False
                detail[backend] = f"运行器文件不存在：{exe}"
                continue
            try:
                result = subprocess.run(cmd + ["--version"], capture_output=True, text=True,
                                        timeout=8, env=_offline_env())
                output = (result.stdout + result.stderr).strip()[:400]
                if result.returncode not in (0, 1):
                    available[backend] = False
                    detail[backend] = f"运行器自检失败（退出码 {result.returncode}）：{output}"
                else:
                    available[backend] = True
                    detail[backend] = output or "自检通过；初始化以冷启动实测为准"
            except (OSError, subprocess.TimeoutExpired) as exc:
                available[backend] = False
                detail[backend] = f"运行器自检异常：{exc}"
        return available, detail


def _on_path(name: str) -> bool:
    import shutil
    return shutil.which(name) is not None


def _offline_env() -> dict[str, str]:
    """§6：推理进程启动时设 HF_HUB_OFFLINE=1，运行时只用本地路径。"""
    import os
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["DOOMSDAY_OFFLINE_CHILD"] = "1"
    return env
