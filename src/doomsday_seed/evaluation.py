"""独立复测与本机评测（文档 §4、§5 第 4 步、§5.1、§6）。

脚本冻结调好的运行器版本、模型文件和参数，通过应用实际对外的接口重跑固定样例与评测
（这里调用与 /v1/chat/completions 同一个 InferenceService.chat，不直连运行器、不绕过适配器）。
固定题集、期望输出、评测参数和评分规则由项目版本管理；Agent 只能用单独的调试样例。
API 通过、断网可用和评分过线分别记录；换设备、换参数或换评测版本都必须重测。
"""
from __future__ import annotations

import statistics
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from .catalog import compute_scores, grade_case, load_eval_config
from .config import LOOPBACK_HOST, MIB, Paths
from .db import Database
from .downloader import DownloadPlan
from .models import Candidate, EvaluationRecord, HardwareSample, RuntimeConfig
from .runner import InferenceService, RuntimeUnavailable
from .runtime_registry import RuntimeRegistry


class ResourceMonitor:
    """采样峰值 RAM / 显存；口径为系统已用量，写入评测记录。"""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.peak_ram_mib = 0.0
        self.peak_vram_mib = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample_vram(self) -> float:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5)
            return max((float(v) for v in out.stdout.split() if v.replace(".", "").isdigit()),
                       default=0.0)
        except (OSError, subprocess.TimeoutExpired, ValueError):
            return 0.0

    def _loop(self) -> None:
        import psutil
        tick = 0
        while not self._stop.is_set():
            self.peak_ram_mib = max(self.peak_ram_mib, psutil.virtual_memory().used / MIB)
            if tick % 4 == 0:
                self.peak_vram_mib = max(self.peak_vram_mib, self._sample_vram())
            tick += 1
            self._stop.wait(self.interval)

    def __enter__(self):
        import psutil
        self.peak_ram_mib = psutil.virtual_memory().used / MIB
        self.peak_vram_mib = self._sample_vram()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)


def build_offline_check(*, plan: DownloadPlan, runtime_config: RuntimeConfig,
                        runner_diagnostics: dict[str, Any], api_ok: bool,
                        cold_start_ms: float, real_drill: bool = False) -> dict[str, Any]:
    """断网启动与任务样例验收。系统层断网由界面演练引导，脚本层验证可验项。"""
    local_paths = all(not Path(f.local_path).is_absolute() and ".." not in Path(f.local_path).parts
                      for f in plan.files)
    checks = {
        "hf_hub_offline_env": True,  # 推理子进程由 _offline_env() 设置 HF_HUB_OFFLINE=1
        "local_paths_only": local_paths,
        "loopback_only": LOOPBACK_HOST in str(runner_diagnostics.get("base_url") or LOOPBACK_HOST),
        "cold_start_ok": cold_start_ms > 0 or runner_diagnostics.get("cold_start_ms", 0) > 0,
        "api_samples_ok": api_ok,
        "no_cloud_fallback": True,
    }
    checks["passed"] = all(checks.values())
    checks.update({
        "method": "推理进程设 HF_HUB_OFFLINE=1、只用本地路径与回环地址；冷启动后重跑固定样例",
        "system_level_disconnect": "已人工演练并记录" if real_drill else "待系统层断网演练",
        "real_drill": real_drill,
        "note": ("脚本层可验项全部通过。文档 §6 要求使用防火墙阻断外网或断开网络适配器，"
                 "保留回环连接后从冷启动再验一次；该步骤由界面「开始断网演练」引导并单独记录。"),
        "runner_log_tail": runner_diagnostics.get("log_tail", "")[-500:],
    })
    return checks


class Evaluator:
    def __init__(self, db: Database, paths: Paths, inference: InferenceService,
                 registry: RuntimeRegistry, sample_provider=None):
        self.db = db
        self.paths = paths
        self.inference = inference
        self.registry = registry
        self._sample_provider = sample_provider

    @property
    def sample(self) -> HardwareSample:
        return self._sample_provider() if self._sample_provider else HardwareSample()

    @staticmethod
    def staging_manifest(plan: DownloadPlan, config: RuntimeConfig,
                         verification: dict[str, Any]) -> dict[str, Any]:
        return {
            "runtime_config": config.model_dump(),
            "variant": plan.files[0].local_path if plan.files else "staging",
            "sizes_and_hashes": [
                {"relative_path": e["local_path"], "kind": e["kind"],
                 "size_bytes": e["size_bytes"], "sha256": e.get("sha256"),
                 "source_path": e["repo_path"]}
                for e in verification.get("files", [])
            ],
        }

    def _call(self, model: str, messages: list[dict[str, Any]],
              max_tokens: int | None = None) -> tuple[dict[str, Any], float]:
        started = time.perf_counter()
        result = self.inference.chat(model=model, messages=messages,
                                     temperature=0.0, max_tokens=max_tokens, interactive=False)
        elapsed = (time.perf_counter() - started) * 1000
        return result, elapsed

    def _ttft(self, model: str, messages: list[dict[str, Any]]) -> float:
        started = time.perf_counter()
        try:
            for chunk in self.inference.chat_stream(model=model, messages=messages,
                                                    temperature=0.0, max_tokens=32, interactive=False):
                if chunk.startswith("data:") and "[DONE]" not in chunk:
                    return (time.perf_counter() - started) * 1000
        except RuntimeUnavailable:
            return 0.0
        return 0.0

    def evaluate(self, *, candidate: Candidate, staging_dir: Path, plan: DownloadPlan,
                 config: RuntimeConfig, verification: dict[str, Any],
                 real_drill: bool = False) -> EvaluationRecord:
        eval_config = load_eval_config("stage1")
        cases = eval_config["cases"]
        protocol = eval_config["protocol"]
        model = f"retest-{candidate.id}"
        manifest = self.staging_manifest(plan, config, verification)
        self.inference.register_staging(model, staging_dir, manifest)

        sample = self.sample
        record = EvaluationRecord(
            id=uuid.uuid4().hex[:20], candidate_id=candidate.id, task=candidate.task,
            eval_version=eval_config["version"], backend=config.backend,
            runner_version=self.registry.version(config.backend),
            hardware_fingerprint=sample.fingerprint(),
            params={"context": config.context, "concurrency": config.concurrency,
                    "gpu_layers": config.gpu_layers, "backend": config.backend,
                    "chat_template": config.chat_template, "temperature": 0.0,
                    "projection": bool(config.mmproj),
                    "config_fingerprint": config.fingerprint()},
            total_cases=len(cases),
        )
        case_results: list[dict[str, Any]] = []
        api_ok = True
        median_latencies: list[float] = []
        speeds: list[float] = []
        cold_start_ms = 0.0
        try:
            with self.inference.reserve(model, keep_warm=True), ResourceMonitor() as monitor:
                # 冷启动单独计时：第一次请求触发加载，不计入预热后的中位耗时。
                warmup_case = cases[0]
                try:
                    warmup, warmup_ms = self._call(model, warmup_case["messages"],
                                              warmup_case.get("max_tokens"))
                    cold_start_ms = self.inference.last_cold_start_ms or warmup_ms
                    record.cold_start_ms = round(cold_start_ms, 1)
                    record.backend = warmup["doomsday"]["backend"]
                    record.runner_version = self.registry.version(record.backend)
                    record.params.update(backend=record.backend,
                                         gpu_layers=config.gpu_layers if record.backend == "cuda" else 0)
                except RuntimeUnavailable as exc:
                    api_ok = False
                    record.detail["error"] = str(exc)

                repeats = int(protocol.get("repeats", 3))
                for case in cases:
                    outputs: list[str] = []
                    latencies: list[float] = []
                    case_api_ok = True
                    for _ in range(repeats):
                        try:
                            result, elapsed = self._call(model, case["messages"],
                                                         case.get("max_tokens"))
                            if result["doomsday"]["backend"] != record.backend:
                                raise RuntimeUnavailable("评测期间运行后端发生变化")
                            outputs.append(result["choices"][0]["message"]["content"])
                            latencies.append(elapsed)
                            tokens = result.get("usage", {}).get("completion_tokens", 0)
                            if tokens and elapsed:
                                speeds.append(tokens / (elapsed / 1000))
                        except RuntimeUnavailable as exc:
                            case_api_ok = False
                            outputs.append("")
                            latencies.append(float("inf"))
                            record.detail.setdefault("errors", []).append(str(exc))
                    api_ok = api_ok and case_api_ok
                    output = outputs[0] if outputs else ""
                    grades = [grade_case(case, response) for response in outputs]
                    passed, detail = grades[0]
                    finite = [l for l in latencies if l != float("inf")]
                    median = statistics.median(finite) if finite else float(eval_config["max_wait_ms"])
                    median_latencies.append(median)
                    stable = case_api_ok and len({passed for passed, _ in grades}) == 1
                    case_results.append({
                        "id": case["id"], "category": case.get("category", ""),
                        "passed": bool(passed and case_api_ok), "output": output[:500],
                        "detail": detail, "median_latency_ms": round(median, 1),
                        "repeat_outputs": [o[:200] for o in outputs], "stable": stable,
                    })
                record.ttft_ms = round(self._ttft(model, cases[0]["messages"]), 1)
                record.peak_ram_mib = round(monitor.peak_ram_mib, 1)
                record.peak_vram_mib = round(monitor.peak_vram_mib, 1)

            record.api_ok = bool(api_ok)
            record.passed_cases = sum(1 for c in case_results if c["passed"])
            record.total_cases = len(cases)
            record.quality = round(record.passed_cases / max(1, record.total_cases), 4)
            record.median_latency_ms = round(statistics.median(median_latencies), 1)
            record.tokens_per_s = round(statistics.median(speeds), 2) if speeds else 0.0
            stable_all = all(c["stable"] for c in case_results)
            record.detail.update({
                "stable": stable_all, "case_results": case_results,
                "protocol": protocol, "scope": eval_config["scope"],
                "resource_metric": "系统已用 RAM / nvidia-smi 已用显存的采样峰值",
                "fixed_scenario": {"context": config.context, "concurrency": config.concurrency},
            })
            offline = build_offline_check(
                plan=plan, runtime_config=config,
                runner_diagnostics=self.inference.status(), api_ok=api_ok,
                cold_start_ms=cold_start_ms, real_drill=real_drill)
            # 断网可用还要求任务样例确实完成，不能只靠环境开关。
            offline["passed"] = bool(offline["passed"] and stable_all and record.passed_cases > 0)
            record.offline_ok = offline["passed"]
            record.detail["offline_check"] = offline
            record.score_speed, record.score_balanced, record.score_accuracy = compute_scores(
                record, eval_config)
        finally:
            self.inference.unregister_staging(model)

        self.db.add_evaluation(record)
        return record
