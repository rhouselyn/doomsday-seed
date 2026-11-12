"""本机探测与资源预算（文档 §3）。

首次启动、每轮更新和加载前采样：CPU 型号/架构/指令集、RAM、GPU 与驱动、
显存总量与可用量、后端能否真正初始化。集成显卡共享内存不计入独显，
多张设备不直接叠加显存。CUDA 初始化失败则走 CPU。
"""
from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, asdict
from functools import lru_cache

from .config import GIB, MIB, Paths, ram_reserve, vram_reserve
from .models import HardwareSample
from .runtime_registry import RuntimeRegistry

# Windows IsProcessorFeaturePresent 特性码（winnt.h）中可稳定核对的一组；
# AVX-512 等更细的特性由运行器启动自检确认，不用易错的位码推断。
_PF_FEATURES = {36: "SSSE3", 37: "SSE4.1", 38: "SSE4.2", 39: "AVX", 40: "AVX2"}


def _instruction_sets() -> tuple[list[str], str]:
    flags: list[str] = []
    source = "unknown"
    if os.name == "nt":
        try:
            import ctypes
            k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            k32.IsProcessorFeaturePresent.argtypes = [ctypes.c_uint]
            k32.IsProcessorFeaturePresent.restype = ctypes.c_bool
            for code, label in _PF_FEATURES.items():
                if k32.IsProcessorFeaturePresent(code):
                    flags.append(label)
            source = "Windows IsProcessorFeaturePresent"
        except Exception:
            source = "Windows 特性探测不可用"
    else:
        try:
            text = open("/proc/cpuinfo", encoding="utf-8", errors="ignore").read()
            line = next((l for l in text.splitlines() if l.lower().startswith("flags")), "")
            known = {"avx", "avx2", "avx512f", "fma", "f16c", "sse4_1", "sse4_2", "bmi2"}
            flags = sorted({f.upper() for f in line.split() if f in known})
            source = "/proc/cpuinfo"
        except OSError:
            source = "指令集信息不可用"
    return sorted(set(flags)), source


@lru_cache(maxsize=1)
def _cpu_model() -> str:
    """优先用 WMI 取可读型号；调用一次即缓存，避免每次采样都启动 PowerShell。"""
    if os.name == "nt":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                 "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)"],
                capture_output=True, text=True, timeout=8)
            name = out.stdout.strip()
            if name:
                return name
        except (OSError, subprocess.TimeoutExpired):
            pass
    return platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER") or "未知 CPU"


def _nvidia_gpus() -> tuple[list[dict], float, float, str]:
    """只读 nvidia-smi；失败时返回明确原因，不臆造显存。"""
    if not shutil.which("nvidia-smi"):
        return [], 0.0, 0.0, "未找到 nvidia-smi，无法确认 NVIDIA 独显与驱动"
    query = "--query-gpu=index,name,driver_version,memory.total,memory.free,memory.used"
    try:
        result = subprocess.run(
            ["nvidia-smi", query, "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=12)
        if result.returncode != 0:
            return [], 0.0, 0.0, f"nvidia-smi 调用失败：{result.stderr.strip()[:200]}"
        gpus: list[dict] = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            gpus.append({
                "index": parts[0], "name": parts[1], "driver_version": parts[2],
                "vram_total_mib": _num(parts[3]), "vram_free_mib": _num(parts[4]),
                "vram_used_mib": _num(parts[5]), "discrete": True,
            })
        if not gpus:
            return [], 0.0, 0.0, "nvidia-smi 未返回 NVIDIA 独显"
        # 多设备不叠加：取空闲最多的一张作为预算基准。
        primary = max(gpus, key=lambda g: g["vram_free_mib"])
        note = f"以 {primary['name']} 为预算基准；多设备不叠加显存"
        if len(gpus) > 1:
            note += f"（检测到 {len(gpus)} 张 NVIDIA 设备）"
        return gpus, primary["vram_total_mib"], primary["vram_free_mib"], note
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [], 0.0, 0.0, f"nvidia-smi 异常：{exc}"


def _num(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def probe(paths: Paths, registry: RuntimeRegistry | None = None) -> HardwareSample:
    import psutil
    vm = psutil.virtual_memory()
    gpus, vram_total, vram_free, gpu_note = _nvidia_gpus()
    driver = gpus[0]["driver_version"] if gpus else ""
    flags, flag_source = _instruction_sets()
    registry = registry or RuntimeRegistry(paths)
    available, detail = registry.probe()
    if not gpus:
        available["cuda"] = False
        detail["cuda"] = gpu_note
    detail["gpu_budget"] = gpu_note
    return HardwareSample(
        os=f"{platform.system()} {platform.release()} ({platform.version()})",
        cpu_model=_cpu_model(), arch=platform.machine(), instruction_sets=flags,
        instruction_source=flag_source, cores=psutil.cpu_count(logical=False) or 0,
        threads=psutil.cpu_count(logical=True) or 0,
        ram_total_mib=vm.total / MIB, ram_available_mib=vm.available / MIB,
        gpus=gpus, vram_total_mib=vram_total, vram_available_mib=vram_free,
        driver_version=driver, backend_available=available, backend_detail=detail,
        runner_version=registry.version(),
    )


@dataclass
class ResourceBudget:
    """§3 预留 max(2 GiB, RAM 10%) 与 max(1 GiB, 显存 10%)。"""

    ram_total_mib: float
    ram_available_mib: float
    ram_reserve_mib: float
    ram_usable_mib: float
    vram_total_mib: float
    vram_available_mib: float
    vram_reserve_mib: float
    vram_usable_mib: float
    backend: str
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def budget(sample: HardwareSample, backend: str = "cpu") -> ResourceBudget:
    r_res = ram_reserve(sample.ram_total_mib * MIB) / MIB
    v_res = vram_reserve(sample.vram_total_mib * MIB) / MIB if sample.vram_total_mib else 0.0
    r_usable = max(0.0, sample.ram_available_mib - r_res)
    v_usable = max(0.0, sample.vram_available_mib - v_res) if sample.vram_total_mib else 0.0
    note = "按当前可用量预留；硬件忙则等待"
    if backend == "cuda" and sample.vram_total_mib == 0:
        note = "无可用独显显存，CUDA 后端不可用，回退 CPU"
    return ResourceBudget(
        ram_total_mib=sample.ram_total_mib, ram_available_mib=sample.ram_available_mib,
        ram_reserve_mib=r_res, ram_usable_mib=r_usable,
        vram_total_mib=sample.vram_total_mib, vram_available_mib=sample.vram_available_mib,
        vram_reserve_mib=v_res, vram_usable_mib=v_usable,
        backend=backend if backend != "cuda" or v_usable > 0 else "cpu", note=note,
    )


def estimate_footprint(weight_bytes: int, context: int = 2048, gpu_layers: int = 0,
                       projection_bytes: int = 0, backend: str = "cpu") -> dict:
    """§3 估算：模型常驻 + KV 缓存/中间张量 + 运行器工作区 + 附件。

    系数是起始规则（可在设置中调整）；只用于「预估」，结论以实测为准。
    """
    weight = weight_bytes / MIB
    attachment = projection_bytes / MIB
    kv = context * 0.0625  # 经验值：短上下文每 token 约 0.0625 MiB
    workspace = 256.0
    offload = max(0.0, min(1.0, gpu_layers / 100.0)) if backend == "cuda" else 0.0
    resident = weight + kv + attachment
    vram = resident * offload + (workspace * 0.25 if offload else 0.0)
    ram = resident * (1 - offload) + workspace
    return {
        "est_ram_mib": round(ram, 1), "est_vram_mib": round(vram, 1),
        "basis": ("固定上下文 {ctx}、并发 1；权重 {w:.0f} MiB + KV {kv:.0f} MiB + "
                  "工作区 {ws:.0f} MiB + 附件 {att:.0f} MiB；GPU 卸载 {off:.0%}").format(
                      ctx=context, w=weight, kv=kv, ws=workspace, att=attachment, off=offload),
        "context": context, "backend": backend,
    }


def check_capacity(estimate: dict, sample: HardwareSample, backend: str) -> dict:
    b = budget(sample, backend)
    ram_ok = b.ram_usable_mib >= estimate.get("est_ram_mib", 0)
    vram_ok = b.vram_usable_mib >= estimate.get("est_vram_mib", 0)
    reasons = []
    if not ram_ok:
        reasons.append(f"内存余量不足（可用 {b.ram_usable_mib:.0f} MiB / 需要 {estimate.get('est_ram_mib', 0):.0f} MiB）")
    if backend == "cuda" and not vram_ok:
        reasons.append(f"显存余量不足（可用 {b.vram_usable_mib:.0f} MiB / 需要 {estimate.get('est_vram_mib', 0):.0f} MiB）")
    return {
        "fits": not reasons, "reason": "；".join(reasons), "budget": b.to_dict(),
        "backend": b.backend,
    }
