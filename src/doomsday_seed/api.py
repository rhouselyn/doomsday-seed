"""本地 HTTP API（文档 §6）。

服务与推理进程只监听回环地址；/api/* 是应用管理接口，/v1/* 是 OpenAI 兼容的
最小聊天接口（含流式）。所有能力状态都可区分「已下载」与「离线可用」。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .catalog import (PROFILE_DEFS, TASKS, load_eval_config, profile_view, rank,
                      task_view, promote, sota_leaders)
from .config import APP_NAME, APP_TAGLINE, APP_VERSION, EVAL_VERSION, LOOPBACK_HOST
from .discovery import LICENSE_ALLOWLIST, local_adapter
from .hardware import budget
from .models import (JOB_STATE_LABELS, PROFILES, STATE_ICONS, STATE_LABELS,
                     ChatCompletionRequest, ChatMessage, DefaultPointer, Settings, utc_now)
from .runner import RuntimeUnavailable
from . import media
from .agent import AgentUnavailable, DeepSeekClient

router = APIRouter()


def get_service(request: Request):
    return request.app.state.service


class SettingsPayload(BaseModel):
    auto_update: bool = True
    schedule_enabled: bool = True
    schedule_time: str = "03:00"
    agent_enabled: bool = True
    agent_base_url: str | None = None
    agent_model: str | None = None
    download_root: str | None = None
    trusted_publishers: list[str] = Field(default_factory=list)
    manual_repos: list[str] = Field(default_factory=list)
    subscriptions: dict[str, list[str]] = Field(default_factory=dict)
    simulated_offline: bool = False


class DownloadPayload(BaseModel):
    candidate_id: str
    task: str = "chat"
    profile: str = "balanced"
    mode: Literal["local", "sota"] = "local"


class PreferencePayload(BaseModel):
    mode: Literal["local", "sota", "off"]
    profiles: list[Literal["speed", "balanced", "accuracy"]] = Field(default_factory=list, max_length=3)
    sota_enabled: bool | None = None


class InferInputs(BaseModel):
    messages: list[ChatMessage] = Field(default_factory=list)
    image: str | None = Field(default=None, max_length=media.IMAGE_BASE64_LIMIT)


class InferPayload(BaseModel):
    task: str
    model: str = Field(min_length=1, max_length=200)
    inputs: InferInputs
    stream: bool = False


class AgentKeyPayload(BaseModel):
    api_key: str = Field(min_length=1, max_length=1024)


class AgentChatPayload(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


class DefaultPayload(BaseModel):
    seed_id: str | None = None
    pinned: bool = True
    auto: bool = False


class ExportPayload(BaseModel):
    out_path: str | None = None
    include_seeds: bool = True


class SimulatePayload(BaseModel):
    enabled: bool = True


class UploadPayload(BaseModel):
    filename: str
    content: str = ""


# ---------------------------------------------------------------- 概览 / 状态
@router.get("/api/health")
def health():
    return {"ok": True, "app": APP_NAME, "version": APP_VERSION,
            "time": utc_now(), "loopback_only": True}


@router.get("/api/session")
def session(service=Depends(get_service)):
    """同源浏览器取得本地会话令牌；其他客户端改用 api-key。"""
    return {"session_token": service.security.session_token,
            "api_key_hint": "本地客户端使用 data/api-key 作为 Bearer 密钥"}


@router.get("/api/status")
def status(service=Depends(get_service)):
    db, paths = service.db, service.paths
    settings = db.get_settings()
    sample = db.latest_sample()
    if sample is None:
        sample = service.hardware()
        db.save_sample(sample)
    source = db.get_source("huggingface")
    online = not settings.simulated_offline and not source.get("last_error")
    seeds = db.list_seeds()
    offline_ready = [s for s in seeds if s.offline_ready and s.integrity == "ok"]
    jobs = db.list_jobs(limit=20)
    return {
        "app": {"name": APP_NAME, "tagline": APP_TAGLINE, "version": APP_VERSION},
        "network": {"online": online,
                    "text": ("外网：在线" if online else
                             "外网不可达。已验证的种子可继续使用。"),
                    "simulated_offline": settings.simulated_offline,
                    "source_checked_at": source.get("last_success"),
                    "source_error": source.get("last_error")},
        "hardware": {
            "cpu": sample.cpu_model, "arch": sample.arch,
            "ram_total_mib": round(sample.ram_total_mib),
            "ram_available_mib": round(sample.ram_available_mib),
            "gpu": (sample.gpus[0]["name"] if sample.gpus else "无 NVIDIA 独显"),
            "vram_total_mib": round(sample.vram_total_mib),
            "vram_available_mib": round(sample.vram_available_mib),
            "backends": sample.backend_available,
            "sampled_at": sample.sampled_at,
        },
        "last_sync": {"attempt": db.get_meta("last_sync_attempt") or None,
                      "success": db.get_meta("last_sync_success") or None,
                      "status": db.get_meta("last_sync_status") or None,
                      "skip_reason": _safe_json(db.get_meta("last_sync_skip_reason"))},
        "counts": {"candidates": len(db.list_candidates()), "seeds": len(seeds),
                   "offline_ready": len(offline_ready),
                   "active_jobs": sum(1 for j in jobs if j.state not in ("done", "failed"))},
        "available_now": [{"task": s.manifest.get("task", "chat"),
                           "variant": s.manifest.get("variant", s.id),
                           "seed_id": s.id} for s in offline_ready],
        "unavailable_reason": ("" if offline_ready else
                               "本机暂无可运行模型：尚无通过评测与断网验收的种子。"
                               "请先下载并完成离线验证。"),
        "inference": service.inference.status(),
        "real_drill": db.get_meta("real_drill") == "1",
    }


def _safe_json(value: str) -> Any:
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return value or None


# ---------------------------------------------------------------- 硬件
@router.get("/api/hardware")
def get_hardware(service=Depends(get_service)):
    sample = service.hardware()
    service.db.save_sample(sample)
    return {
        "sampled_at": sample.sampled_at, "sample": sample.model_dump(),
        "budget_cpu": budget(sample, "cpu").to_dict(),
        "budget_cuda": budget(sample, "cuda").to_dict(),
        "runner_registry": service.registry.load(),
        "fingerprint": sample.fingerprint(),
    }


# ---------------------------------------------------------------- 目录
@router.get("/api/catalog")
def catalog(task: str | None = None, profile: str | None = None,
            state: str | None = None, mode: Literal["local", "sota"] = "local",
            service=Depends(get_service)):
    db = service.db
    if task and task not in TASKS:
        raise HTTPException(400, f"未知能力：{task}")
    if profile and profile not in PROFILES:
        raise HTTPException(400, f"未知档位：{profile}")
    rankings = sota_leaders(db.list_candidates(task=task))
    candidates = db.list_candidates(task=task, state=state)
    jobs = db.list_jobs(limit=200)
    items = []
    for c in candidates:
        evals = db.list_evaluations(candidate_id=c.id)
        seed = db.find_seed_by_candidate(c.id)
        if mode == "sota" and c.id not in rankings and not (seed and seed.manifest.get("collection_mode") == "sota"):
            continue
        related_jobs = [j for j in jobs if j.candidate_id == c.id][:5]
        latest = evals[0] if evals else None
        items.append({
            **c.model_dump(),
            "state_label": STATE_LABELS.get(c.state, c.state),
            "state_icon": STATE_ICONS.get(c.state, ""),
            "install_block": (
                "模型需要在线授权" if c.state == "gated" else
                "许可尚未确认或不符合下载规则" if c.license not in LICENSE_ALLOWLIST else
                "此版本暂不支持本机试用" if not local_adapter(c) else ""
            ),
            "collectable": c.id in rankings and c.state != "gated" and c.license in LICENSE_ALLOWLIST,
            "sota": rankings.get(c.id, []),
            "public_evidence": [e.model_dump() for e in c.evidence],
            "estimate": {**c.estimate.model_dump(), "label": "预估"},
            "measured": ([{
                "quality": latest.quality, "passed_cases": latest.passed_cases,
                "total_cases": latest.total_cases, "ttft_ms": latest.ttft_ms,
                "median_latency_ms": latest.median_latency_ms,
                "tokens_per_s": latest.tokens_per_s, "peak_ram_mib": latest.peak_ram_mib,
                "peak_vram_mib": latest.peak_vram_mib, "backend": latest.backend,
                "eval_version": latest.eval_version, "created_at": latest.created_at,
                "offline_ok": latest.offline_ok,
            }] if latest else []),
            "measured_label": "本机实测" if latest else "待测",
            "seed": ({"id": seed.id, "completeness": seed.completeness,
                      "offline_ready": seed.offline_ready, "integrity": seed.integrity,
                      "rel_dir": seed.rel_dir,
                      "collection_mode": seed.manifest.get("collection_mode", "local")} if seed else None),
            "jobs": [{"id": j.id, "state": j.state,
                      "state_label": JOB_STATE_LABELS.get(j.state, j.state),
                      "progress": j.progress, "message": j.message} for j in related_jobs],
            "download_progress": (max((j.progress for j in related_jobs), default=0.0)),
        })
    defaults = []
    for pointer in db.list_defaults():
        seed = db.get_seed(pointer.seed_id) if pointer.seed_id else None
        defaults.append({**pointer.model_dump(),
                         "variant": seed.manifest.get("variant") if seed else None,
                         "offline_ready": bool(seed and seed.offline_ready)})
    return {
        "tasks": task_view(), "profiles": profile_view(),
        "eval_version": EVAL_VERSION,
        "candidates": items, "defaults": defaults,
        "filter": {"task": task, "profile": profile, "state": state, "mode": mode},
        "note": ("页面分别展示公开证据、兼容性预估和本机实测，并附测试时间与覆盖范围；"
                 "一个档位的「领先」只表示在已实测候选、相同任务和本机评测配置下领先。"),
    }


@router.get("/api/seeds/{seed_id}")
def seed_detail(seed_id: str, service=Depends(get_service)):
    db, paths = service.db, service.paths
    seed = db.get_seed(seed_id)
    if seed is None:
        raise HTTPException(404, f"种子不存在：{seed_id}")
    directory = paths.seeds / seed.rel_dir
    usage = ""
    if (directory / "USAGE.md").exists():
        usage = (directory / "USAGE.md").read_text(encoding="utf-8")
    candidate = db.get_candidate(seed.candidate_id)
    history = [{"id": s.id, "created_at": s.created_at,
                "offline_ready": s.offline_ready, "completeness": s.completeness}
               for s in db.list_seeds()
               if s.manifest.get("variant") == seed.manifest.get("variant")]
    return {
        "seed": seed.model_dump(), "candidate": candidate.model_dump() if candidate else None,
        "usage_markdown": usage, "history": history,
        "package_files": seed.manifest.get("sizes_and_hashes", []),
        "offline_verification": seed.manifest.get("offline_verification", {}),
        "evaluation": seed.manifest.get("evaluation"),
        "runner": seed.manifest.get("runner", {}),
    }


@router.get("/api/evaluations")
def evaluations(candidate_id: str | None = None, service=Depends(get_service)):
    return {"items": [e.model_dump() for e in service.db.list_evaluations(candidate_id)]}


@router.get("/api/agent/config")
def agent_config(service=Depends(get_service)):
    settings = service.db.get_settings()
    client = service.gateway.client
    return {"configured": bool(service.credentials.get()),
            "provider": getattr(client, "name", "local-rule-agent"),
            "model": getattr(client, "model", "") or settings.agent_model or DeepSeekClient.default_model,
            "base_url": getattr(client, "base_url", "") or settings.agent_base_url}


@router.put("/api/agent/config")
def save_agent_key(payload: AgentKeyPayload, service=Depends(get_service)):
    try:
        service.credentials.set(payload.api_key)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(400, str(exc)) from exc
    settings = service.db.get_settings()
    settings.agent_enabled = True
    service.db.put_settings(settings)
    service.gateway.client = DeepSeekClient(service.credentials,
                                           base_url=settings.agent_base_url,
                                           model=settings.agent_model)
    service.gateway.enabled = True
    return agent_config(service)


@router.delete("/api/agent/config")
def delete_agent_key(service=Depends(get_service)):
    service.credentials.clear()
    service.gateway.client = service.build_agent_client()
    return agent_config(service)


@router.post("/api/agent/test")
def test_agent_connection(service=Depends(get_service)):
    settings = service.db.get_settings()
    client = DeepSeekClient(service.credentials, base_url=settings.agent_base_url,
                            model=settings.agent_model)
    try:
        reply = client.complete([
            {"role": "user", "content": "请只回复 OK。"},
        ])
        return {"connected": bool(reply.get("content")), "provider": "deepseek",
                "model": client.model}
    except AgentUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc


@router.get("/api/agent/history")
def agent_history(service=Depends(get_service)):
    return {"messages": _safe_json(service.db.get_meta("agent:conversation")) or []}


@router.delete("/api/agent/history")
def clear_agent_history(service=Depends(get_service)):
    service.db.set_meta("agent:conversation", "[]")
    return {"messages": []}


@router.post("/api/agent/chat")
def agent_chat(payload: AgentChatPayload, service=Depends(get_service)):
    from .agent_interaction import chat
    try:
        return chat(service, payload.message.strip())
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except AgentUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc


@router.get("/api/agent/discoveries")
def get_agent_discoveries(service=Depends(get_service)):
    from .researcher import latest
    return {"report": latest(service)}


@router.post("/api/agent/discoveries")
def run_agent_discovery(service=Depends(get_service)):
    from .researcher import discover
    return {"report": discover(service)}


# ---------------------------------------------------------------- 设置与计划
@router.get("/api/settings")
def get_settings(service=Depends(get_service)):
    settings = service.db.get_settings()
    return {**settings.model_dump(), "effective": service.scheduler.status(),
            "seeds_dir": str(service.paths.seeds),
            "default_seeds_dir": str(service.paths.default_seeds),
            "agent_mode": "rules" if service.gateway.client.name == "local-rule-agent" else "remote"}


def _move_library(service, target: Path) -> None:
    """归档根目录切换：现有模型一并移动，避免模型库忽然变空。"""
    paths = service.paths
    source = paths.seeds.resolve()
    target = target.resolve()
    if source == target:
        return
    if source in target.parents or target in source.parents:
        raise HTTPException(400, "新目录不能与当前归档目录互相包含")
    active = [j for j in service.db.list_jobs(limit=200)
              if j.state not in ("done", "failed")]
    if active:
        raise HTTPException(409, "仍有下载或验证任务在进行，请完成后再修改保存位置")
    target.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        if (target / child.name).exists():
            raise HTTPException(409, f"目标目录已存在同名项目：{child.name}")
    import shutil
    for child in source.iterdir():
        shutil.move(str(child), str(target / child.name))
    paths.set_download_root(str(target))


@router.put("/api/settings")
def put_settings(payload: SettingsPayload, service=Depends(get_service)):
    db, paths = service.db, service.paths
    old = db.get_settings()
    for task, profiles in payload.subscriptions.items():
        if task not in TASKS:
            raise HTTPException(400, f"未知能力：{task}")
        for p in profiles:
            if p not in PROFILES:
                raise HTTPException(400, f"未知档位：{p}")
    subscriptions = payload.subscriptions or old.subscriptions
    base_url = old.agent_base_url if payload.agent_base_url is None else payload.agent_base_url.strip()
    model_name = old.agent_model if payload.agent_model is None else payload.agent_model.strip()
    download_root = old.download_root if payload.download_root is None else payload.download_root.strip()
    if base_url and not base_url.lower().startswith(("http://", "https://")):
        raise HTTPException(400, "调用链接需要以 http:// 或 https:// 开头")
    try:
        target = paths.resolve_download_root(download_root)
        _move_library(service, target)
    except HTTPException:
        raise
    except (OSError, ValueError) as exc:
        raise HTTPException(400, f"保存位置不可用：{exc}") from exc
    effective_root = "" if target == paths.default_seeds.resolve() else str(target)
    settings = Settings(auto_update=payload.auto_update,
                        schedule_enabled=payload.schedule_enabled,
                        schedule_time=payload.schedule_time,
                        agent_enabled=payload.agent_enabled,
                        agent_base_url=base_url, agent_model=model_name,
                        download_root=effective_root,
                        subscriptions=subscriptions,
                        download_modes=old.download_modes,
                        sota_subscriptions=old.sota_subscriptions,
                        trusted_publishers=payload.trusted_publishers,
                        manual_repos=payload.manual_repos,
                        simulated_offline=payload.simulated_offline)
    saved = db.put_settings(settings)
    service.gateway.enabled = saved.agent_enabled
    service.reload_agent_client()
    effective = service.scheduler.status()
    if (old.schedule_time != saved.schedule_time
            or old.schedule_enabled != saved.schedule_enabled
            or (saved.schedule_enabled and not effective.get("registered"))):
        effective = service.scheduler.apply_settings(saved) | service.scheduler.status()
    return {**saved.model_dump(), "effective": effective,
            "seeds_dir": str(paths.seeds), "default_seeds_dir": str(paths.default_seeds),
            "agent_mode": "rules" if service.gateway.client.name == "local-rule-agent" else "remote",
            "note": "返回的是实际生效状态；任务计划注册失败时以 effective 为准"}


@router.get("/api/preferences")
def preferences(service=Depends(get_service)):
    settings = service.db.get_settings()
    return {task: {"mode": settings.download_modes.get(task, "local" if task in settings.subscriptions else "off"),
                   "profiles": settings.subscriptions.get(task, ["balanced"]),
                   "sota_enabled": task in settings.sota_subscriptions or settings.download_modes.get(task) == "sota"}
            for task in TASKS}


@router.put("/api/preferences/{task}")
def set_preference(task: str, payload: PreferencePayload, service=Depends(get_service)):
    if task not in TASKS:
        raise HTTPException(404, "模型类型不存在")
    if payload.mode == "local" and not payload.profiles:
        raise HTTPException(400, "本机自动收录至少需要一个档位")
    settings = service.db.get_settings()
    settings.download_modes[task] = payload.mode
    settings.subscriptions[task] = list(dict.fromkeys(payload.profiles))
    if payload.sota_enabled is not None:
        settings.sota_subscriptions = [t for t in settings.sota_subscriptions if t != task]
        if payload.sota_enabled:
            settings.sota_subscriptions.append(task)
    service.db.put_settings(settings)
    return preferences(service)[task]


@router.get("/api/schedule")
def schedule_status(service=Depends(get_service)):
    return service.scheduler.status()


@router.post("/api/schedule/register")
def schedule_register(service=Depends(get_service)):
    settings = service.db.get_settings()
    return service.scheduler.register(settings.schedule_time)


@router.delete("/api/schedule/register")
def schedule_unregister(service=Depends(get_service)):
    return service.scheduler.unregister()


# ---------------------------------------------------------------- 同步与作业
@router.post("/api/sync")
def sync(trigger: str = Query("manual"), wait: bool = Query(False), catalog_only: bool = Query(False),
         service=Depends(get_service)):
    report = service.pipeline.sync(trigger=trigger, wait=wait, catalog_only=catalog_only)
    return report.model_dump()


@router.post("/api/downloads")
def create_download(payload: DownloadPayload, service=Depends(get_service)):
    try:
        job = service.pipeline.submit_download(payload.candidate_id, payload.task, payload.profile, mode=payload.mode)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"job_id": job.id, "state": job.state, "idempotency_key": job.idempotency_key}


@router.get("/api/jobs")
def list_jobs(state: str | None = None, service=Depends(get_service)):
    jobs = service.db.list_jobs(limit=200, state=state)
    return {"items": [j.model_dump() for j in jobs]}


@router.get("/api/jobs/{job_id}")
def get_job(job_id: str, service=Depends(get_service)):
    job = service.db.get_job(job_id)
    if job is None:
        raise HTTPException(404, f"作业不存在：{job_id}")
    return job.model_dump()


# ---------------------------------------------------------------- 默认版本
@router.put("/api/defaults/{task}/{profile}")
def put_default(task: str, profile: str, payload: DefaultPayload,
                service=Depends(get_service)):
    db = service.db
    if task not in TASKS or profile not in PROFILES:
        raise HTTPException(400, "能力或档位不存在")
    pointer = db.get_default(task, profile)
    old_seed = pointer.seed_id
    if payload.auto:
        # 恢复自动推荐时保留上个可用版本。
        pointer.pinned = False
        pointer.source = "auto"
        pointer.previous_seed_id = old_seed
        config = load_eval_config()
        records = [e for e in db.list_evaluations(eval_version=config["version"])]
        candidates = []
        for record in records:
            seed = db.find_seed_by_candidate(record.candidate_id)
            if record.task == task and seed and seed.offline_ready and seed.integrity == "ok":
                candidates.append((record, seed))
        if candidates:
            ranked = rank([r for r, _ in candidates], profile)
            best = ranked[0]
            best_seed = next(s for r, s in candidates if r.id == best.id)
            ok, reason = promote(None, best, profile, config)
            if ok:
                pointer.seed_id = best_seed.id
            db.set_meta(f"auto_recommend:{task}:{profile}", reason)
        db.set_default(pointer)
    elif payload.seed_id:
        seed = db.get_seed(payload.seed_id)
        if (seed is None or not seed.offline_ready or seed.integrity != "ok"
                or seed.manifest.get("task", "chat") != task):
            raise HTTPException(400, "只能固定或切换到「离线可用」的已验证版本")
        pointer.seed_id = seed.id
        pointer.pinned = bool(payload.pinned)
        pointer.previous_seed_id = old_seed
        pointer.source = "user"
        db.set_default(pointer)
    else:
        raise HTTPException(400, "请提供 seed_id 或使用 auto=true")
    return {"default": db.get_default(task, profile).model_dump(),
            "previous_seed_id": old_seed}


# ---------------------------------------------------------------- 本地文本附件
@router.post("/api/uploads")
def validate_text_upload(payload: UploadPayload, service=Depends(get_service)):
    """§6 上传限制文件类型与尺寸；内容不外发、不落盘、不保留试用输入。"""
    from .config import ALLOWED_TEXT_SUFFIXES, MAX_UPLOAD_BYTES
    from .security import validate_upload
    raw = payload.content.encode("utf-8")
    validate_upload(payload.filename, len(raw))
    return {
        "filename": payload.filename, "size": len(raw),
        "text": payload.content[:20000], "retained": False, "local_only": True,
        "allowed_suffixes": list(ALLOWED_TEXT_SUFFIXES), "max_bytes": MAX_UPLOAD_BYTES,
    }


# ---------------------------------------------------------------- 断网演练与恢复
@router.get("/api/drill")
def drill_status(service=Depends(get_service)):
    return service.recovery.status()


@router.post("/api/drill/simulate")
def drill_simulate(payload: SimulatePayload, service=Depends(get_service)):
    return service.recovery.set_simulated(payload.enabled)


@router.post("/api/drill/record")
def drill_record(service=Depends(get_service)):
    return service.recovery.record_real_drill()


@router.post("/api/export")
def export_package(payload: ExportPayload, service=Depends(get_service)):
    return service.recovery.export(payload.out_path, payload.include_seeds)


@router.post("/api/export/wheelhouse")
def build_wheelhouse(service=Depends(get_service)):
    return service.recovery.build_wheelhouse()


# ---------------------------------------------------------------- OpenAI 兼容
@router.get("/v1/models")
def v1_models(service=Depends(get_service)):
    return {"object": "list", "data": service.inference.models()}


@router.post("/api/infer")
def infer(req: InferPayload, service=Depends(get_service)):
    task = TASKS.get(req.task)
    if not task:
        raise HTTPException(404, "模型类型不存在")
    if not task["enabled"]:
        raise HTTPException(501, "此类型的本地试用尚未接入")
    if req.task == "chat":
        seed = service.db.get_seed(req.model)
        if seed and seed.manifest.get("task", "chat") != "chat":
            raise HTTPException(400, "模型与任务类型不匹配")
        response = v1_chat(ChatCompletionRequest(model=req.model, messages=req.inputs.messages,
                                                  stream=req.stream), service)
        if req.stream:
            return response
        return {"task": req.task, "model": req.model,
                "outputs": {"text": response["choices"][0]["message"]["content"]},
                "runtime": response.get("doomsday", {})}
    if req.stream or not req.inputs.image:
        raise HTTPException(422, "关键点识别需要一张图片")
    seed = service.db.get_seed(req.model)
    if not seed or not seed.offline_ready or seed.integrity != "ok":
        raise HTTPException(409, "此模型尚未完成本机验证")
    if seed.manifest.get("task") != req.task or seed.manifest.get("runner", {}).get("adapter") != "mediapipe":
        raise HTTPException(400, "模型与任务类型不匹配")
    try:
        picture = media.decode_image(req.inputs.image)
        entry = next(item for item in seed.manifest["sizes_and_hashes"] if item["kind"] == "weight")
        from .debug_tools import safe_path
        weight = safe_path(service.paths.seeds / seed.rel_dir, entry["relative_path"])
        outputs = media.detect(service.paths, req.task, weight, picture)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(503, "本地运行环境暂不可用") from exc
    return {"task": req.task, "model": req.model, "outputs": outputs,
            "runtime": {"backend": "cpu", "version": media.VERSION}}


@router.post("/v1/chat/completions")
def v1_chat(req: ChatCompletionRequest, service=Depends(get_service)):
    messages = [m.model_dump() for m in req.messages]
    if not messages:
        raise HTTPException(400, {"error": {"message": "messages 不能为空"}})
    try:
        if req.stream:
            def stream_events():
                try:
                    yield from service.inference.chat_stream(
                        req.model, messages, temperature=req.temperature,
                        max_tokens=req.max_tokens)
                except RuntimeUnavailable as exc:
                    yield "data: " + json.dumps(
                        {"error": {"message": str(exc), "code": exc.code}},
                        ensure_ascii=False) + "\n\n"

            return StreamingResponse(
                stream_events(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        return service.inference.chat(
            req.model, messages, temperature=req.temperature, max_tokens=req.max_tokens)
    except RuntimeUnavailable as exc:
        raise HTTPException(
            status_code=404 if exc.code == "no_model" else 503,
            detail={"error": {"message": str(exc), "type": "local_only_no_cloud_fallback",
                              "code": exc.code}}) from exc


# ---------------------------------------------------------------- 后续能力占位
@router.post("/api/ocr")
def ocr_placeholder():
    """§6：任务接口随能力接入；阶段 2 才登记运行器与评测。"""
    raise HTTPException(
        status_code=501,
        detail={"error": "文字识别（OCR）属于阶段 2，尚未接入本机运行器与评测；"
                         "当前不提供该能力，也不会转发到云端",
                "phase": 2, "task": "ocr"})
