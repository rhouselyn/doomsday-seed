"""Read-only model discovery and its saved report."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json

from .agent import AgentContext, LocalRuleAgent
from .config import HF_MAX_WORKERS
from .models import utc_now


RESEARCH_REPOS = {
    "chat": "Qwen/Qwen3-8B",
    "code": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "math": "Qwen/Qwen2.5-Math-7B-Instruct",
    "translation": "Helsinki-NLP/opus-mt-zh-en",
    "vision": "Qwen/Qwen3-VL-8B-Instruct",
    "ocr": "PaddlePaddle/PaddleOCR-VL-1.5",
    "image_gen": "stabilityai/stable-diffusion-xl-base-1.0",
    "image_edit": "Qwen/Qwen-Image-Edit",
    "speech": "openai/whisper-large-v3",
    "tts": "hexgrad/Kokoro-82M",
    "video_gen": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "video": "Qwen/Qwen2.5-VL-3B-Instruct",
    "retrieval": "BAAI/bge-m3",
}


def discover(service) -> dict:
    """Fetch metadata only. This path never schedules jobs or fetches model files."""
    api = service.source.api()

    def lookup(item):
        task, repo = item
        try:
            info = api.model_info(repo, files_metadata=True, timeout=20)
            if not info.sha:
                raise ValueError("上游未提供固定版本")
            models = service.source._to_candidates(info)
            for model in models:
                model.task = task
            return task, repo, info, models, ""
        except Exception as exc:
            return task, repo, None, [], f"{type(exc).__name__}: {str(exc)[:160]}"

    rows = []
    with ThreadPoolExecutor(max_workers=HF_MAX_WORKERS) as pool:
        for task, repo, info, models, error in pool.map(lookup, RESEARCH_REPOS.items()):
            if error:
                rows.append({"task": task, "repo": repo, "status": "来源暂不可达", "error": error})
                continue
            for model in models:
                service.db.upsert_candidate(model)
            items = [{
                "id": m.id, "repo": m.repo, "commit_rev": m.commit_rev,
                "variant": m.variant, "task": m.task, "files": [f.repo_path for f in m.files],
                "backend": m.estimate.backend, "estimate_basis": m.estimate.basis,
                "est_ram_mib": m.estimate.est_ram_mib, "evidence_count": len(m.evidence),
                "eligible": True,
            } for m in models]
            selected = LocalRuleAgent().recommend(AgentContext(candidates=items, max_suggestions=1))
            rows.append({
                "task": task, "repo": repo, "revision": info.sha, "status": "已读取元数据",
                "url": f"https://huggingface.co/{repo}/tree/{info.sha}",
                "variant": selected[0].variant if selected else None,
                "note": "公开来源候选，尚未证明 SOTA 或本机可运行" if selected else "当前格式未提取为可下载清单",
            })
    for candidate in service.db.list_candidates():
        if candidate.source == "mediapipe":
            rows.append({"task": candidate.task, "repo": candidate.repo, "revision": candidate.commit_rev,
                         "status": "官方模型已收录", "url": candidate.evidence[0].url,
                         "variant": candidate.variant, "note": "本机状态以模型库为准"})
    rows.append({"task": "face_swap", "repo": "facefusion/facefusion-assets",
                 "status": "需核对许可与来源",
                 "url": "https://github.com/facefusion/facefusion-assets",
                 "note": "换脸模型候选尚未确定可归档的权重与许可"})
    report = {"created_at": utc_now(), "method": "公开模型元数据与本地规则",
              "downloaded": False, "items": rows}
    raw_history = service.db.get_meta("agent:discovery_history")
    try:
        history = json.loads(raw_history) if raw_history else []
        if not isinstance(history, list):
            history = []
    except ValueError:
        history = []
    service.db.set_meta("agent:discovery_history", json.dumps((history + [report])[-10:], ensure_ascii=False))
    service.db.set_meta("agent:discovery", json.dumps(report, ensure_ascii=False))
    return report


def latest(service) -> dict | None:
    raw = service.db.get_meta("agent:discovery")
    try:
        return json.loads(raw) if raw else None
    except ValueError:
        return None
