"""Agent 受控调试工具（文档 §5.1）。

工具只对当前作业的暂存区和回环服务开放。Agent 接触的是兼容性所需的硬件摘要与诊断，
不持有下载凭据、正式种子库写权限或默认版本切换权限；来源网页、模型卡和日志按资料处理。
工具校验相对路径、参数和启动程序，只调用已安装的受控运行器。需要新运行器、
第三方代码或全新接口时，作业标为「待适配」。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .adapters import registered_adapters
from .agent import AgentGateway
from .catalog import load_debug_samples, grade_case
from .config import FIXED_CONTEXT, FIXED_CONCURRENCY, Paths
from .db import Database
from .downloader import DownloadPlan, Downloader
from .models import RuntimeConfig
from .runner import InferenceService, RunningServer, RuntimeUnavailable
from .runtime_registry import RuntimeRegistry

ALLOWED_TEMPLATES = {None, "", "chatml", "qwen", "llama3", "gemma", "default", "auto"}
GPU_LAYERS_RANGE = (0, 100)
CONTEXT_RANGE = (512, 8192)


class DebugPermissionError(Exception):
    pass


def safe_path(base: Path, relative: str) -> Path:
    """只允许当前作业暂存区内的相对路径。"""
    if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise DebugPermissionError(f"非法相对路径：{relative}")
    resolved = (base / relative).resolve()
    if not resolved.is_relative_to(base.resolve()):
        raise DebugPermissionError(f"路径越出暂存区：{relative}")
    return resolved


class DebugToolbox:
    def __init__(self, db: Database, paths: Paths, downloader: Downloader,
                 inference: InferenceService, registry: RuntimeRegistry):
        self.db = db
        self.paths = paths
        self.downloader = downloader
        self.inference = inference
        self.registry = registry

    # ---------- 只读资料 ----------
    def read_manifest(self, job_id: str) -> dict[str, Any]:
        base = self.downloader.staging_dir(job_id).resolve()
        path = safe_path(base, "plan.json")
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def read_diagnostics(self, job_id: str) -> dict[str, Any]:
        base = self.downloader.staging_dir(job_id).resolve()
        path = safe_path(base, "diagnostics.json")
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def write_diagnostics(self, job_id: str, data: dict[str, Any]) -> None:
        base = self.downloader.staging_dir(job_id).resolve()
        base.mkdir(parents=True, exist_ok=True)
        safe_path(base, "diagnostics.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def list_adapters() -> list[dict[str, str]]:
        return registered_adapters()

    # ---------- 参数与启动程序校验 ----------
    def validate_config(self, config: dict[str, Any]) -> RuntimeConfig:
        if config.get("adapter") not in {a["id"] for a in registered_adapters()}:
            raise DebugPermissionError("适配器未登记：作业应标「待适配」")
        if config.get("backend") not in ("cpu", "cuda"):
            raise DebugPermissionError("后端不在允许范围内")
        layers = int(config.get("gpu_layers", 0))
        context = int(config.get("context", FIXED_CONTEXT))
        if not GPU_LAYERS_RANGE[0] <= layers <= GPU_LAYERS_RANGE[1]:
            raise DebugPermissionError("GPU 层数超出允许范围")
        if not CONTEXT_RANGE[0] <= context <= CONTEXT_RANGE[1]:
            raise DebugPermissionError("上下文长度超出允许范围")
        if int(config.get("concurrency", FIXED_CONCURRENCY)) != FIXED_CONCURRENCY:
            raise DebugPermissionError("首版固定并发 1，不能修改")
        template = config.get("chat_template")
        if template not in ALLOWED_TEMPLATES:
            raise DebugPermissionError(f"对话模板不在白名单：{template}")
        command = self.registry.command(config["backend"])
        if not command:
            raise DebugPermissionError("启动程序未登记，只调用已安装的受控运行器")
        exe = Path(command[0])
        if not exe.exists() and not any(part in str(exe) for part in ("python", "mock")):
            raise DebugPermissionError(f"启动程序不可执行：{exe}")
        return RuntimeConfig(**{**config, "verified": False})

    def submit_config(self, job_id: str, config: RuntimeConfig) -> dict[str, Any]:
        """Agent 只提交可校验的运行配置；写入暂存区，不触碰种子库与默认版本。"""
        base = self.downloader.staging_dir(job_id).resolve()
        base.mkdir(parents=True, exist_ok=True)
        data = config.model_dump()
        data["alternate_configuration"] = config.context != FIXED_CONTEXT
        if data["alternate_configuration"]:
            data["note"] = "与固定 2048 token 场景不同，须标成另一种试用配置并单独测试"
        safe_path(base, "tuned_config.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return data

    # ---------- 暂存服务与样例请求 ----------
    def _staging_manifest(self, plan: DownloadPlan, config: RuntimeConfig) -> dict[str, Any]:
        return {
            "runtime_config": config.model_dump(),
            "variant": plan.files[0].local_path if plan.files else "staging",
            "sizes_and_hashes": [
                {"relative_path": f.local_path, "kind": f.kind, "size_bytes": f.size_bytes,
                 "sha256": f.sha256, "source_path": f.repo_path}
                for f in plan.files
            ],
        }

    def _start(self, job_id: str, plan: DownloadPlan, config: RuntimeConfig,
               backend: str | None = None) -> RunningServer:
        files_dir = self.downloader.files_dir(job_id)
        self.inference.register_staging(f"staging-{job_id}", files_dir,
                                        self._staging_manifest(plan, config))
        return self.inference.start_staging_server(f"staging-{job_id}", config, backend)

    def sample_request(self, server: RunningServer, case: dict[str, Any]) -> dict[str, Any]:
        try:
            result = self.inference.chat_on_server(
                server, case["messages"], max_tokens=case.get("max_tokens"))
            content = result["choices"][0]["message"]["content"]
            passed, detail = grade_case(case, content)
            return {"id": case["id"], "ok": passed, "output": content, "detail": detail,
                    "api_ok": True}
        except Exception as exc:
            return {"id": case["id"], "ok": False, "output": "", "api_ok": False,
                    "detail": f"{type(exc).__name__}: {exc}"}

    def run_debug(self, job_id: str, plan: DownloadPlan, base_config: RuntimeConfig,
                  gateway: AgentGateway) -> tuple[RuntimeConfig, dict[str, Any]]:
        """启动暂存服务 → 调试样例 → 诊断 → Agent 在允许范围内调参 → 再验。"""
        diagnostics: dict[str, Any] = {
            "job_id": job_id, "adapter": base_config.adapter,
            "adapter_registered": True, "attempts": [], "samples": [],
            "agent_note": "", "scope": "仅当前作业暂存区与回环服务",
        }
        config = base_config.model_copy(deep=True)
        server: RunningServer | None = None
        try:
            config = self.validate_config(config.model_dump())
            for attempt in range(2):
                try:
                    server = self._start(job_id, plan, config)
                    diagnostics["attempts"].append({
                        "attempt": attempt + 1, "backend": server.backend,
                        "cold_start_ms": server.cold_start_ms,
                        "log_tail": server.diagnostics()["log_tail"][-500:],
                    })
                    break
                except RuntimeUnavailable as exc:
                    if config.backend == "cuda" and ("cuda" in str(exc).lower()
                                                     or exc.code == "runtime"):
                        diagnostics["cuda_init_failed"] = True
                        config.backend = "cpu"
                        config.gpu_layers = 0
                        continue
                    diagnostics["error"] = str(exc)
                    diagnostics["pending_adaptation"] = True
                    raise
            if server is None:
                raise RuntimeUnavailable("暂存服务两次尝试均失败", "runtime")

            for case in load_debug_samples():
                diagnostics["samples"].append(self.sample_request(server, case))
            failed = [s for s in diagnostics["samples"] if not s["ok"]]
            if failed:
                diagnostics["template_mismatch"] = all(
                    "包含" in s["detail"] or "JSON" in s["detail"] for s in failed)
                suggested, note = gateway.tune({
                    "runtime_config": config.model_dump(),
                    "diagnostics": {
                        "cuda_init_failed": diagnostics.get("cuda_init_failed", False),
                        "template_mismatch": diagnostics.get("template_mismatch", False),
                        "sample_errors": [s["detail"] for s in failed],
                    },
                })
                diagnostics["agent_note"] = note
                if suggested:
                    merged = {**config.model_dump(), **{
                        k: v for k, v in suggested.items()
                        if k in RuntimeConfig.model_fields}}
                    self.inference.stop_server(server)
                    server = None
                    config = self.validate_config(merged)
                    server = self._start(job_id, plan, config)
                    diagnostics["retry_samples"] = [
                        self.sample_request(server, case) for case in load_debug_samples()]
                    failed = [s for s in diagnostics["retry_samples"] if not s["ok"]]
            diagnostics["debug_passed"] = not failed
            diagnostics["api_ok"] = all(s.get("api_ok") for s in diagnostics["samples"])
            config.verified = not failed
            self.submit_config(job_id, config)
            return config, diagnostics
        except DebugPermissionError as exc:
            diagnostics["error"] = str(exc)
            diagnostics["pending_adaptation"] = True
            diagnostics["debug_passed"] = False
            return config, diagnostics
        except RuntimeUnavailable as exc:
            diagnostics.setdefault("error", str(exc))
            diagnostics["pending_adaptation"] = exc.code == "unregistered"
            diagnostics["debug_passed"] = False
            return config, diagnostics
        finally:
            if server is not None:
                self.inference.stop_server(server)
            self.inference.unregister_staging(f"staging-{job_id}")
            self.write_diagnostics(job_id, diagnostics)
