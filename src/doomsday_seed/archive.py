"""归档、清单、离线说明与启动对账（文档 §5 第 5 步、§5.1、§6）。

每个包写 manifest.json（来源、固定版本、变体、相对路径、大小与哈希、许可、
运行器版本、运行参数、评测和断网验证记录）与 USAGE.md（归档启动方法、
最小 API 请求、测试范围和限制）。完整度按实际文件与字节数计算。
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .config import Paths
from .db import Database
from .downloader import sha256_file
from .models import Candidate, DownloadPlan, EvaluationRecord, HardwareSample, Job, RuntimeConfig, utc_now

MANIFEST_VERSION = "1.0"


def build_manifest(*, seed_id: str, candidate: Candidate, plan: DownloadPlan,
                   verification: dict[str, Any], runtime_config: RuntimeConfig,
                   evaluation: EvaluationRecord | None, hardware: HardwareSample,
                   offline_check: dict[str, Any],
                   external_references: list[str] | None = None) -> dict[str, Any]:
    """§5.5 清单字段；命令与指标只能来自已验证配置与测试记录。"""
    files = []
    for entry in verification.get("files", []):
        files.append({
            "relative_path": entry["local_path"], "kind": entry["kind"],
            "size_bytes": entry["size_bytes"], "sha256": entry.get("sha256"),
            "upstream_sha256": entry.get("upstream_sha256"),
            "source_path": entry["repo_path"],
        })
    runner_cmd = runner_command(runtime_config, plan, seed_id)
    return {
        "manifest_version": MANIFEST_VERSION,
        "seed_id": seed_id,
        "candidate_id": candidate.id,
        "created_at": utc_now(),
        "source": {
            "name": candidate.source, "repo": candidate.repo,
            "fixed_revision": plan.commit_rev, "publisher": candidate.publisher,
            "upstream_model_card": next((f.local_path for f in plan.files
                                         if f.kind == "model_card"), None),
            "license": candidate.license,
            "license_file": next((f.local_path for f in plan.files
                                  if f.kind == "license"), None),
            "usage_docs": [f.local_path for f in plan.files if f.kind == "usage"],
            "third_party_quantization": {
                "original_model": candidate.repo,
                "conversion_publisher": candidate.publisher,
                "note": "第三方量化包记录原始模型与转换发布者",
            },
        },
        "variant": candidate.variant,
        "task": candidate.task,
        "relative_paths": [f["relative_path"] for f in files],
        "sizes_and_hashes": files,
        "total_bytes": verification.get("total_bytes", 0),
        "runner": {
            "name": "llama.cpp llama-server", "version": runtime_config.runner_version,
            "backend": runtime_config.backend, "adapter": runtime_config.adapter,
            "command": runner_cmd,
        },
        "runtime_config": runtime_config.model_dump(),
        "evaluation": (evaluation.model_dump() if evaluation else None),
        "offline_verification": offline_check,
        "hardware": {
            "cpu_model": hardware.cpu_model, "arch": hardware.arch,
            "gpus": [g.get("name") for g in hardware.gpus],
            "driver_version": hardware.driver_version,
            "ram_total_mib": hardware.ram_total_mib,
            "vram_total_mib": hardware.vram_total_mib,
            "fingerprint": hardware.fingerprint(),
        },
        "external_references": external_references if external_references is not None else [
            f"https://huggingface.co/{candidate.repo}/tree/{plan.commit_rev}",
        ],
        "notes": "必要的外链资料按许可保存；未随包保存的链接在此标明",
    }


def runner_command(config: RuntimeConfig, plan: DownloadPlan, seed_id: str) -> list[str]:
    """从已验证的运行配置生成启动命令；模型一律用本地路径。"""
    weight = next((f.local_path for f in plan.files if f.kind == "weight"), None)
    cmd = ["llama-server", "--host", "127.0.0.1", "--port", "8177",
           "-c", str(config.context), "-np", str(config.concurrency)]
    if weight:
        cmd += ["-m", f"seeds/{seed_id}/{weight}"]
    if config.backend == "cuda":
        cmd += ["-ngl", str(config.gpu_layers)]
    if config.mmproj:
        cmd += ["--mmproj", f"seeds/{seed_id}/{config.mmproj}"]
    if config.chat_template:
        cmd += ["--chat-template", config.chat_template]
    cmd += config.extra_args
    return cmd


def usage_markdown(*, manifest: dict[str, Any], candidate: Candidate,
                   evaluation: EvaluationRecord | None,
                   offline_check: dict[str, Any]) -> str:
    """§5.1：用复测实际通过的配置和样例生成；Agent 只可整理文字。"""
    cfg = manifest["runner"]
    variant = manifest["variant"]
    cmd = " ".join(cfg["command"])
    quality = f"{evaluation.quality:.2f}（{evaluation.passed_cases}/{evaluation.total_cases} 个固定用例通过）" if evaluation else "未取得"
    latency = f"{evaluation.median_latency_ms:.0f} ms" if evaluation else "未取得"
    ttft = f"{evaluation.ttft_ms:.0f} ms" if evaluation else "未取得"
    tps = f"{evaluation.tokens_per_s:.1f} tokens/s" if evaluation else "未取得"
    expected = "（见评测记录中的固定用例与期望输出）"
    if evaluation and evaluation.detail.get("case_results"):
        first = evaluation.detail["case_results"][0]
        expected = f"`{str(first.get('output', ''))[:120]}`"
    return f"""# {variant} · 离线使用说明

> 本文件由复测实际通过的配置与测试记录生成，命令和指标不是人工估计。

## 来源与版本

- 源仓库：`{candidate.repo}`
- 固定修订：`{manifest['source']['fixed_revision']}`
- 变体：`{variant}`
- 许可：`{candidate.license or '未确认'}`
- 上游模型卡 / 用法文档：{', '.join('`' + str(p) + '`' for p in manifest['source']['usage_docs']) or '随包保存于模型卡文件'}
- 第三方量化包：原始模型 `{candidate.repo}`，转换发布者 `{candidate.publisher}`

## 本机验证过的环境

- CPU：{manifest['hardware']['cpu_model']}（{manifest['hardware']['arch']}）
- GPU / 驱动：{', '.join(manifest['hardware']['gpus']) or '无独显'} / {manifest['hardware']['driver_version'] or '不适用'}
- 运行器：`{cfg['name']}` 版本 `{cfg['version']}`，后端 `{cfg['backend']}`
- 硬件指纹：`{manifest['hardware']['fingerprint']}`（换设备或换驱动需重新验证）

## 启动命令

```bash
{cmd}
```

运行时只使用本地路径；推理进程启动前设置 `HF_HUB_OFFLINE=1`，加载器使用本地模式。

## 最小 API 请求

```bash
curl http://127.0.0.1:8777/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{{"model": "{manifest['seed_id']}", "messages": [{{"role": "user", "content": "你好"}}]}}'
```

流式请求加 `"stream": true`。最小预期输出：{expected}

## 可用的输入规格

- 上下文：{manifest['runtime_config']['context']} token；并发：{manifest['runtime_config']['concurrency']}
- 对话模板：`{manifest['runtime_config'].get('chat_template') or '运行器默认'}`
- 首版只接收文本消息与本地文本文件；图片理解、OCR 等能力将在后续阶段接入。

## 实测速度与质量

- 评测配置版本：`{evaluation.eval_version if evaluation else '未取得'}`
- 质量：{quality}
- 端到端中位耗时：{latency}；首 token：{ttft}；生成速度：{tps}
- 峰值 RAM / 显存：{f"{evaluation.peak_ram_mib:.0f} / {evaluation.peak_vram_mib:.0f} MiB" if evaluation else '未取得'}

这些成绩只说明上述固定用例、固定参数与本机配置下的表现；公开原始精度成绩不得挪给量化版，
跨任务也不混排。打开更长上下文、更大图片或换设备需重新验证。

## 断网验证

- 状态：{'通过' if offline_check.get('passed') else '未通过'}
- 方式：{offline_check.get('method', '未记录')}
- 说明：{offline_check.get('note', '')}

## 已知限制

- 单次只加载一个模型；正在交互时后台评测排队。
- 断网后不会回退到云端；找不到可用模型时返回可读错误。
- 未随包保存的外链资料：{', '.join(manifest['external_references']) or '无'}。
"""


def write_manifest(seed_dir: Path, manifest: dict[str, Any]) -> Path:
    path = seed_dir / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_usage(seed_dir: Path, markdown: str) -> Path:
    path = seed_dir / "USAGE.md"
    path.write_text(markdown, encoding="utf-8")
    return path


def compute_completeness(seed_dir: Path, manifest: dict[str, Any]) -> tuple[float, list[str]]:
    """按实际文件与字节数计算完整度，返回 (0–1, 缺失说明)。"""
    declared = manifest.get("sizes_and_hashes", [])
    if not declared:
        return 0.0, ["清单中没有文件"]
    expected_bytes = sum(int(f.get("size_bytes") or 0) for f in declared)
    present_bytes = 0
    missing: list[str] = []
    for item in declared:
        path = seed_dir / item["relative_path"]
        if not path.exists():
            missing.append(f"缺少必需文件：{item['relative_path']}")
            continue
        size = path.stat().st_size
        if item.get("size_bytes") and size != item["size_bytes"]:
            missing.append(f"{item['relative_path']} 字节数不符（期望 {item['size_bytes']}，实际 {size}）")
        present_bytes += size
    ratio = min(1.0, present_bytes / max(1, expected_bytes)) if expected_bytes else 1.0
    return round(ratio, 4), missing


def verify_seed(seed_dir: Path, manifest: dict[str, Any]) -> tuple[bool, list[str], list[str]]:
    """返回 (是否完好, 缺件错误, 哈希不符)；缺件错误可定位到具体文件。"""
    missing: list[str] = []
    mismatched: list[str] = []
    for item in manifest.get("sizes_and_hashes", []):
        path = seed_dir / item["relative_path"]
        if not path.is_file():
            missing.append(f"缺少必需文件：{item['relative_path']}（源路径 {item.get('source_path', '?')}）")
            continue
        if item.get("size_bytes") and path.stat().st_size != item["size_bytes"]:
            missing.append(f"{item['relative_path']} 字节数不符")
        digest = item.get("sha256")
        if digest and sha256_file(path).lower() != str(digest).lower():
            mismatched.append(f"{item['relative_path']} SHA-256 不符")
    return not missing and not mismatched, missing, mismatched


def reconcile(db: Database, paths: Paths) -> dict[str, Any]:
    """§5 启动时对账：接回有效包，残缺包继续排队。"""
    report: dict[str, Any] = {"attached": [], "incomplete": [], "requeued": []}
    paths.seeds.mkdir(parents=True, exist_ok=True)

    # 磁盘上有清单但数据库没有记录 → 接回有效包。
    known = {s.rel_dir for s in db.list_seeds()}
    for directory in sorted(paths.seeds.iterdir()):
        manifest_path = directory / "manifest.json"
        if not directory.is_dir() or not manifest_path.exists() or directory.name in known:
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        candidate_id = manifest.get("candidate_id") or db.get_meta(f"candidate-of:{directory.name}", "")
        from .models import Seed
        seed = Seed(id=directory.name, candidate_id=candidate_id, rel_dir=directory.name,
                    manifest=manifest, completeness=0.0,
                    offline_ready=bool(manifest.get("offline_verification", {}).get("passed")))
        db.put_seed(seed)
        report["attached"].append(directory.name)

    for seed in db.list_seeds():
        directory = paths.seeds / seed.rel_dir
        manifest = seed.manifest
        if not manifest and (directory / "manifest.json").exists():
            try:
                manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                manifest = {}
        ok, missing, mismatched = verify_seed(directory, manifest)
        ratio, _ = compute_completeness(directory, manifest)
        seed.completeness = ratio
        seed.manifest = manifest
        if ok and (directory / "USAGE.md").exists():
            seed.integrity = "ok"
        else:
            seed.integrity = "incomplete"
            seed.offline_ready = False
            report["incomplete"].append({"seed_id": seed.id,
                                         "errors": (missing + mismatched)[:5]})
            candidate = db.get_candidate(seed.candidate_id)
            db.set_candidate_state(seed.candidate_id, "downloaded",
                                   "文件已下载，正在验证离线启动。")
            if candidate:
                job = Job(id=f"recover-{seed.id}", kind="install",
                          candidate_id=candidate.id,
                          idempotency_key=candidate.idempotency_key,
                          state="queued", stage="排队",
                          message="启动对账发现残缺包，继续排队按同一版本续跑")
                _, created = db.create_job(job)
                if created:
                    report["requeued"].append(job.id)
        db.put_seed(seed)
    return report
