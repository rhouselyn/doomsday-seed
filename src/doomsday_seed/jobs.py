"""作业状态机与同步编排（文档 §5、§8）。

目录与作业分开记录。作业走 排队 → 下载 → 校验 → 调试 → 复测 → 完成，
异常进入「等待网络」「等待资源」或「失败」。完整度按实际文件与字节数计算。
重启后从 SQLite 持久化作业恢复；`来源 + 仓库 + commit + 变体` 用作下载幂等键。
默认版本是「能力 + 档位」的单独映射，只在复测与断网验收都通过后按档位规则切换。
"""
from __future__ import annotations

import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Callable

from .agent import AgentGateway
from .archive import (build_manifest, compute_completeness, usage_markdown, verify_seed,
                      write_manifest, write_usage)
from .catalog import load_eval_config, meets_gates, promote, sota_leaders
from .config import Paths
from .db import Database
from .debug_tools import DebugToolbox
from .discovery import HuggingFaceSource, LICENSE_ALLOWLIST, local_adapter
from .downloader import DownloadError, DownloadPlan, Downloader
from .evaluation import Evaluator
from .locks import LockBusy, file_lock
from .hardware import check_capacity, estimate_footprint, probe as probe_hardware
from .models import (PROFILES, Candidate, EvaluationRecord, FileSpec, HardwareSample, Job, RuntimeConfig,
                     Seed, Settings, SyncReport, utc_now)
from .preselect import PreSelector, ShortlistResult
from .runner import InferenceService, RuntimeUnavailable
from .runtime_registry import RuntimeRegistry
from . import media
from .researcher import refresh_reviews


class SyncBusy(Exception):
    pass


@dataclass
class JobResult:
    job_id: str
    state: str
    message: str
    seed_id: str | None = None
    evaluation: EvaluationRecord | None = None


class JobPipeline:
    def __init__(self, db: Database, paths: Paths, *, registry: RuntimeRegistry,
                 source: HuggingFaceSource, preselector: PreSelector,
                 downloader: Downloader, debug_box: DebugToolbox,
                 evaluator: Evaluator, inference: InferenceService,
                 gateway: AgentGateway):
        self.db = db
        self.paths = paths
        self.registry = registry
        self.source = source
        self.preselector = preselector
        self.downloader = downloader
        self.debug_box = debug_box
        self.evaluator = evaluator
        self.inference = inference
        self.gateway = gateway
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='seed-jobs')
        self._last_report: SyncReport | None = None

    def hardware(self) -> HardwareSample:
        return probe_hardware(self.paths, self.registry)

    # ---------- 同步 ----------
    def sync(self, trigger: str = "manual", wait: bool = False, catalog_only: bool = False) -> SyncReport:
        report = SyncReport(id=uuid.uuid4().hex[:16], trigger=trigger)
        self.db.start_sync(report)
        try:
            with file_lock(self.paths.sync_lock):
                settings = self.db.get_settings()
                if settings.simulated_offline:
                    report.status = "skipped"
                    report.message = "模拟模式已暂停同步"
                    report.skipped.append({"reason": report.message})
                    self.db.set_meta("last_sync_attempt", utc_now())
                    self.db.set_meta("last_sync_skip_reason", report.message)
                    return report
                self.db.set_meta("last_sync_attempt", utc_now())
                curated = catalog_only or not settings.auto_update or trigger in {"scheduled", "catchup"}
                result = self.source.discover_library(settings) if curated else self.source.discover(settings)
                result.candidates.extend(media.candidates())
                if curated:
                    self.preselector.store_discovery(result, settings)
                    if self.source.supports_public_reviews:
                        refresh_reviews(self.db)
                    report.discovered = len(result.candidates)
                    sample = self.hardware()
                    self.db.save_sample(sample)
                    models = self.db.list_candidates()
                    slots = self.preselector.recommendations(models, sample, settings, sota_leaders(models))
                    report.shortlisted = list(dict.fromkeys(
                        candidate_id for choices in slots.values() for candidate_id in choices.values()))
                    recommended_ids = set(report.shortlisted)
                    self.source.cache_model_cards([c for c in models if c.id in recommended_ids])
                    current = self.db.get_settings()
                    if current.auto_update and current.schedule_enabled and not current.simulated_offline and not catalog_only:
                        report.jobs = self._queue_recommendations(slots, current, report)
                    report.status = "partial" if result.errors else "ok"
                    self.db.set_meta("last_sync_status", report.status)
                    if not result.errors:
                        self.db.set_meta("last_sync_success", utc_now())
                    self._start_worker(report.jobs, wait=wait)
                    return report
                local_pairs = settings.active_pairs()
                if any(task == "chat" for task, _ in local_pairs):
                    try:
                        self.registry.prepare_cpu()
                    except Exception:
                        report.skipped.append({"reason": "对话环境暂未准备好，仍会更新目录"})
                sample = self.hardware() if local_pairs else HardwareSample()
                if local_pairs:
                    self.db.save_sample(sample)
                if result.retained_previous:
                    report.skipped.append({"reason": "来源不可达，保留旧目录并使用受控候选",
                                           "detail": "; ".join(e["error"] for e in result.errors)[:300]})
                shortlist = self.preselector.build(result, sample, settings,
                                                   registry_version=self.registry.version())
                report.discovered = len(result.candidates)
                report.shortlisted = [p.candidate_id for p in shortlist.plans]
                report.skipped.extend(shortlist.skipped)
                report.message = shortlist.agent_note
                job_ids = self._enqueue(shortlist, report.id)
                models = self.db.list_candidates()
                eligible = [c for c in models if c.state != "gated" and c.license in LICENSE_ALLOWLIST]
                rankings = sota_leaders(models)
                sota_tasks = set(settings.sota_subscriptions)
                sota_tasks.update(task for task, mode in settings.download_modes.items() if mode == "sota")
                for task in sota_tasks:
                    choices = [c for c in eligible if c.task == task and c.id in rankings
                               and not self.db.find_seed_by_candidate(c.id)]
                    if choices:
                        winner = min(choices, key=lambda c: (-len(rankings[c.id]), c.id))
                        job = self.submit_download(winner.id, task, mode="sota", autostart=False)
                        if job.state == "queued":
                            job_ids.append(job.id)
                for task, profile in dict(local_pairs).items():
                    if task not in media.ASSETS:
                        continue
                    entry = next(c for c in media.candidates() if c.task == task)
                    if not self.db.find_seed_by_candidate(entry.id):
                        job = self.submit_download(entry.id, task, profile, autostart=False)
                        if job.state == "queued":
                            job_ids.append(job.id)
                for job in self.db.list_jobs(limit=200):
                    if job.state not in ("waiting_network", "waiting_resource") or not job.payload.get("plan"):
                        continue
                    self.db.update_job(job.id, state="queued", stage="排队", message="自动重试等待中的作业")
                    job_ids.append(job.id)
                job_ids = list(dict.fromkeys(job_ids))
                report.jobs = job_ids
                report.status = "ok" if not result.errors else "partial"
                self.db.set_meta("last_sync_attempt", utc_now())
                self.db.set_meta("last_sync_status", report.status)
                self.db.set_meta("last_sync_skip_reason", json.dumps(report.skipped[:5], ensure_ascii=False))
                if not job_ids and report.status == "ok":
                    self.db.set_meta("last_sync_success", utc_now())
                self._start_worker(job_ids, wait=wait, report=report)
        except (SyncBusy, LockBusy) as exc:
            report.status = "skipped"
            report.message = str(exc)
            report.skipped.append({"reason": str(exc)})
            self.db.set_meta("last_sync_attempt", utc_now())
            self.db.set_meta("last_sync_skip_reason", str(exc))
        except Exception as exc:
            report.status = "failed"
            report.message = f"{type(exc).__name__}: {exc}"
            report.skipped.append({"reason": report.message})
        finally:
            report.finished_at = utc_now()
            self.db.finish_sync(report)
            self._last_report = report
        return report

    def _queue_recommendations(self, slots: dict[str, dict[str, str]], settings: Settings,
                               report: SyncReport) -> list[str]:
        selected = [(task, profile) for task, profile in settings.active_pairs()]
        selected.extend((task, "sota") for task in slots
                        if task in settings.sota_subscriptions or settings.download_modes.get(task) == "sota")
        existing = {seed.candidate_id for seed in self.db.list_seeds()}
        existing.update(job.candidate_id for job in self.db.list_jobs(limit=-1))
        jobs = []
        for task, profile in selected:
            candidate_id = slots.get(task, {}).get(profile)
            if not candidate_id or candidate_id in existing:
                continue
            existing.add(candidate_id)
            candidate = self.db.get_candidate(candidate_id)
            mode = "local" if local_adapter(candidate) else "requested"
            if profile == "sota":
                mode = "sota"
            try:
                job = self.submit_download(candidate_id, task, "balanced" if profile == "sota" else profile,
                                           mode=mode, autostart=False)
            except ValueError as exc:
                report.skipped.append({"candidate_id": candidate_id, "reason": str(exc)})
                continue
            if job.state == "queued":
                jobs.append(job.id)
        return jobs

    def _enqueue(self, shortlist: ShortlistResult, sync_id: str) -> list[str]:
        job_ids: list[str] = []
        for plan in shortlist.plans:
            candidate = self.db.get_candidate(plan.candidate_id)
            if candidate is None:
                continue
            assignments = [a for a in shortlist.assignments
                           if a["candidate_id"] == candidate.id]
            job = Job(
                id=uuid.uuid4().hex[:16], kind="install", candidate_id=candidate.id,
                idempotency_key=candidate.idempotency_key, state="queued", stage="排队",
                progress=0.0, message="已入围，等待下载",
                payload={"plan": plan.model_dump(), "assignments": assignments,
                         "sync_id": sync_id},
            )
            created_job, created = self.db.create_job(job)
            if created:
                job_ids.append(created_job.id)
            else:
                # 跨档 / 跨轮命中同一幂等键：只引用同一个作业。
                if created_job.id not in job_ids:
                    job_ids.append(created_job.id)
        return job_ids

    # ---------- 后台执行 ----------
    def _start_worker(self, job_ids: list[str], wait: bool, report: SyncReport | None = None) -> None:
        if not job_ids:
            return
        if wait:
            self._run_jobs(job_ids, report)
        else:
            self._worker.submit(self._run_jobs, job_ids, report)

    def shutdown(self) -> None:
        self._worker.shutdown(wait=True, cancel_futures=True)

    @contextmanager
    def library_lock(self, *, wait=False):
        with file_lock(self.paths.data / "library.lock", wait=wait):
            configured = self.db.get_settings().download_root
            if configured != self.paths.download_root:
                self.paths.set_download_root(configured)
            yield

    def _run_jobs(self, job_ids: list[str], report: SyncReport | None = None) -> None:
        # 单次只加载一个模型：作业串行执行，后台评测由资源锁排队。
        results = []
        for job_id in job_ids:
            try:
                results.append(self.run_job(job_id))
            except Exception as exc:
                results.append(self._fail(job_id, "failed", str(exc), None))
        if report:
            incomplete = [r for r in results if r.state != "done"]
            if incomplete:
                report.status = "partial" if len(incomplete) < len(results) else "failed"
                report.message = "；".join(r.message for r in incomplete)[:1000]
            elif report.status == "ok":
                self.db.set_meta("last_sync_success", utc_now())
            report.finished_at = utc_now()
            self.db.set_meta("last_sync_status", report.status)
            self.db.finish_sync(report)

    def recover(self) -> list[str]:
        try:
            with self.library_lock():
                jobs = self.db.recover_jobs()
        except LockBusy:
            jobs = self.db.list_jobs(state="queued", limit=-1)
        return [j.id for j in jobs]

    def run_pending(self) -> None:
        if self.db.get_settings().simulated_offline:
            return
        job_ids = self.recover()
        if job_ids:
            self._start_worker(job_ids, wait=False)

    def submit_download(self, candidate_id: str, task: str = "chat",
                        profile: str = "balanced", autostart: bool = True, mode: str = "local") -> Job:
        if self.db.get_settings().simulated_offline:
            raise ValueError("模拟模式已暂停新下载")
        candidate = self.db.get_candidate(candidate_id)
        if candidate is None:
            raise ValueError(f"候选不存在：{candidate_id}")
        if candidate.state == "gated":
            raise ValueError("该模型为「待授权」，请先在线办理授权")
        if task != candidate.task or profile not in PROFILES or mode not in {"local", "sota", "requested"}:
            raise ValueError("所选能力或档位暂不可用")
        if candidate.license not in LICENSE_ALLOWLIST:
            raise ValueError("许可尚未确认或不符合下载规则")
        try:
            candidate = self.source.pin(candidate)
        except Exception as exc:
            raise ValueError(f"无法确认固定版本的文件清单：{exc}") from exc
        if candidate.state == "gated" or candidate.task != task or candidate.license not in LICENSE_ALLOWLIST:
            raise ValueError("固定版本的能力、访问权限或许可不符合下载规则")
        adapter = local_adapter(candidate)
        if mode == "local" and not adapter:
            raise ValueError("固定版本暂不支持本机试用")
        if mode == "sota" and candidate.id not in sota_leaders(self.db.list_candidates(task=task)):
            raise ValueError("固定版本没有可用于 SOTA 收藏的领先记录")
        config = RuntimeConfig(adapter=adapter or "archive-only",
                               mmproj=next((f.local_path for f in candidate.files if f.kind == "projection"), None))
        plan = DownloadPlan(candidate_id=candidate.id, repo=candidate.repo,
                            commit_rev=candidate.commit_rev, files=candidate.files,
                            runtime_config=config, total_bytes=candidate.size_bytes)
        job = Job(id=uuid.uuid4().hex[:16], kind="install", candidate_id=candidate.id,
                  idempotency_key=candidate.idempotency_key, state="queued", stage="排队",
                  payload={"plan": plan.model_dump(),
                           "assignments": [{"task": task, "profile": profile,
                                            "candidate_id": candidate.id,
                                            "reason": "用户手动下载"}],
                           "sync_id": "manual", "mode": mode})
        with self.db.transaction():
            previous = self.db.query_one("SELECT state FROM jobs WHERE idempotency_key=?",
                                         (job.idempotency_key,))
            retry = bool(previous and previous["state"] in
                         ("failed", "waiting_network", "waiting_resource"))
            created, _ = self.db.create_job(job)
            saved = self.db.find_seed_by_candidate(candidate.id)
            upgrade = created.state == "done" and mode == "local" and saved and not saved.offline_ready
            if retry or upgrade:
                self.db.update_job(created.id, state="queued", stage="排队", progress=0.0,
                                   message="等待下载和验证", payload=job.payload)
                created = self.db.get_job(created.id)
        if created.state == "queued" and autostart:
            self._start_worker([created.id], wait=False)
        return created

    # ---------- 单个作业状态机 ----------
    @staticmethod
    def _progress(job_id: str, db: Database) -> Callable[[float, str], None]:
        def update(fraction: float, message: str) -> None:
            db.update_job(job_id, progress=round(min(0.6, 0.05 + fraction * 0.55), 3),
                          message=message)
        return update

    def run_job(self, job_id: str) -> JobResult:
        with self.library_lock(wait=True):
            return self._run_job(job_id)

    def _run_job(self, job_id: str) -> JobResult:
        job = self.db.get_job(job_id)
        if job is None or job.state != "queued":
            return JobResult(job_id, job.state if job else "missing", "作业无需执行")
        if job.kind != "local_import" and self.db.get_settings().simulated_offline:
            return JobResult(job_id, "queued", "模拟模式已暂停下载")
        claim = self.db.execute("UPDATE jobs SET state='downloading', stage='下载', updated_at=? "
                                "WHERE id=? AND state='queued'", (utc_now(), job_id))
        if not claim.rowcount:
            return JobResult(job_id, "downloading", "作业已由其他进程接手")
        if job.kind == "local_import":
            from .local_import import run
            return run(self, job)
        try:
            plan = DownloadPlan(**job.payload["plan"])
        except (KeyError, TypeError, ValueError) as exc:
            return self._fail(job_id, "failed", f"下载计划无效：{exc}", None)
        assignments = job.payload.get("assignments", [])
        archive_only = job.payload.get("mode") in {"sota", "requested"}
        candidate = self.db.get_candidate(plan.candidate_id)
        if candidate is None:
            self.db.update_job(job_id, state="failed", stage="失败",
                               message="候选记录缺失，作业终止")
            return JobResult(job_id, "failed", "候选记录缺失")
        try:
            # 排队 → 下载
            self.db.update_job(job_id, state="downloading", stage="下载", progress=0.05,
                               message="下载预检：固定 commit、精确文件清单、落盘空间")
            if candidate.source == "huggingface":
                try:
                    fixed = self.source.pin(candidate, plan.commit_rev)
                except Exception as exc:
                    raise DownloadError(f"无法确认固定版本的文件清单：{exc}", "network") from exc
                if fixed.state == "gated" or fixed.license not in LICENSE_ALLOWLIST:
                    raise DownloadError("固定版本的访问权限或许可不符合规则", "other")
                if fixed.task != candidate.task:
                    raise DownloadError("固定版本的任务类型已改变，请重新选择", "other")
                if fixed.id != candidate.id:
                    existing = self.db.query_one("SELECT id FROM jobs WHERE idempotency_key=?", (fixed.idempotency_key,))
                    if existing and existing["id"] != job_id:
                        duplicate = self.db.get_job(existing["id"])
                        message = f"固定版本已有作业：{duplicate.id}"
                        state = "done" if duplicate.state == "done" else "failed"
                        self.db.update_job(job_id, state=state, stage=state, message=message,
                                           payload={**job.payload, "replacement_job": duplicate.id})
                        return JobResult(job_id, state, message)
                    candidate = fixed
                    plan.candidate_id = fixed.id
                    plan.commit_rev = fixed.commit_rev
                    plan.files = fixed.files
                    plan.runtime_config.mmproj = next((f.local_path for f in plan.files if f.kind == "projection"), None)
                    job.payload["plan"] = plan.model_dump()
                    with self.db.transaction():
                        self.db.execute("UPDATE jobs SET candidate_id=?, idempotency_key=? WHERE id=?",
                                        (fixed.id, fixed.idempotency_key, job_id))
                        self.db.update_job(job_id, payload=job.payload)
                if fixed.files != plan.files:
                    plan.files = fixed.files
                    plan.runtime_config.mmproj = next((f.local_path for f in plan.files if f.kind == "projection"), None)
                    job.payload["plan"] = plan.model_dump()
                    self.db.update_job(job_id, payload=job.payload)
            archived = self.db.find_seed_by_candidate(candidate.id)
            reuse = archived and archived.integrity == "ok" and archived.manifest["source"]["fixed_revision"] == plan.commit_rev
            if reuse:
                plan.files = [FileSpec(repo_path=f["source_path"], local_path=f["relative_path"],
                                       size_bytes=f["size_bytes"], kind=f["kind"],
                                       sha256=f.get("sha256"), upstream_sha256=f.get("upstream_sha256"))
                              for f in archived.manifest["sizes_and_hashes"]]
            preflight = self.downloader.preflight(plan, job_id, reuse=bool(reuse))
            if not preflight.ok:
                raise DownloadError("；".join(preflight.reasons), "resource")
            staging = self.downloader.staging_dir(job_id)
            staging.mkdir(parents=True, exist_ok=True)
            (staging / "plan.json").write_text(
                json.dumps(plan.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
            files_dir = (self.paths.seeds / archived.rel_dir if reuse else
                         self.downloader.download(plan, job_id, self._progress(job_id, self.db)))
            self.db.set_candidate_state(candidate.id, "downloaded",
                                        "文件已下载，正在验证离线启动。")

            # 校验
            self.db.update_job(job_id, state="verifying", stage="校验", progress=0.65,
                               message="校验文件大小、上游摘要与本地 SHA-256")
            verification = self.downloader.verify(files_dir, plan)
            if archive_only or local_adapter(candidate) == "mediapipe":
                ready = not archive_only
                if ready:
                    self.db.update_job(job_id, state="debugging", stage="准备", progress=0.72,
                                       message="正在准备本地动作模型环境")
                    try:
                        media.prepare(self.paths)
                    except Exception as exc:
                        raise DownloadError("动作环境暂不可用，已保留权重", "network") from exc
                    with self.inference.reserve(f"smoke-{job_id}"):
                        media.smoke(self.paths, candidate.task, files_dir / candidate.variant)
                return self._store_simple(job, candidate, plan, files_dir, verification, ready, bool(reuse))

            # 调试（Agent 只操作暂存区与回环服务）
            self.db.update_job(job_id, state="debugging", stage="调试", progress=0.72,
                               message="正在准备本地对话环境")
            if plan.runtime_config.runner_version == "unregistered":
                try:
                    self.registry.prepare_cpu()
                except Exception as exc:
                    if not self.registry.command("cuda"):
                        raise DownloadError("对话环境暂不可用，已保留权重", "network") from exc
                current = self.hardware()
                self.db.save_sample(current)
                backend = "cuda" if current.backend_available.get("cuda") else "cpu"
                plan.runtime_config.backend = backend
                plan.runtime_config.runner_version = self.registry.version(backend)
                plan.runtime_config.gpu_layers = 99 if backend == "cuda" else 0
                plan.runtime_config.mmproj = next((f.local_path for f in plan.files if f.kind == "projection"), None)
                weight_bytes = sum(f.size_bytes for f in plan.files if f.kind == "weight")
                projection_bytes = sum(f.size_bytes for f in plan.files if f.kind == "projection")
                estimate = estimate_footprint(weight_bytes, projection_bytes=projection_bytes,
                                               backend=backend, gpu_layers=plan.runtime_config.gpu_layers)
                capacity = check_capacity(estimate, current, backend)
                if not capacity["fits"]:
                    raise DownloadError(capacity["reason"], "resource")
            tuned, diagnostics = self.debug_box.run_debug(
                job_id, plan, plan.runtime_config, self.gateway, files_dir)
            if diagnostics.get("pending_adaptation"):
                reason = f"待适配：{diagnostics.get('error', '缺少受控运行器或适配器')}"
                self.db.set_candidate_state(candidate.id, "unsupported", reason)
                raise RuntimeUnavailable(reason, "unregistered")
            if not diagnostics.get("debug_passed"):
                raise RuntimeUnavailable(
                    f"调试样例未通过：{diagnostics.get('error') or '见 diagnostics.json'}", "runtime")

            # 复测（正式 API 固定题集 + 断网启动）
            self.db.update_job(job_id, state="retesting", stage="复测", progress=0.8,
                               message="通过应用正式接口冷启动复测固定样例与评测")
            evaluation = self.evaluator.evaluate(
                candidate=candidate, staging_dir=files_dir, plan=plan, config=tuned,
                verification=verification,
                real_drill=self.db.get_meta("real_drill") == "1")

            # 归档：写清单与离线说明 → 重命名 → 事务更新可见状态
            self.db.update_job(job_id, state="retesting", stage="复测", progress=0.92,
                               message="写入 manifest.json 与 USAGE.md")
            plan.runtime_config = tuned
            seed, _ = self._save_evaluation(candidate, plan, files_dir, verification,
                                                       evaluation, None if reuse else job_id)
            offline_ready = seed.offline_ready
            promoted = self._update_defaults(candidate.id, evaluation, assignments) if offline_ready else []
            message = ("离线可用，已归档" if offline_ready else
                       "复测未过线，已归档为「已下载 · 待验证」")
            if promoted:
                message += "；默认版本：" + "、".join(promoted)
            self.db.update_job(job_id, state="done", stage="完成", progress=1.0,
                               message=message,
                               payload={**job.payload, "seed_id": candidate.id,
                                        "evaluation_id": evaluation.id,
                                        "diagnostics": diagnostics})
            return JobResult(job_id, "done", message, candidate.id, evaluation)

        except DownloadError as exc:
            state = {"network": "waiting_network", "resource": "waiting_resource"}.get(
                exc.code, "failed")
            return self._fail(job_id, state, f"{exc}", candidate)
        except RuntimeUnavailable as exc:
            state = "waiting_resource" if exc.code == "waiting_resource" else "failed"
            return self._fail(job_id, state, f"{exc}", candidate)
        except Exception as exc:
            return self._fail(job_id, "failed", f"{type(exc).__name__}: {exc}", candidate)
        finally:
            # 成功归档后清理暂存区的 raw 缓存；失败时保留现场便于按同一版本续跑。
            current = self.db.get_job(job_id)
            if current and current.state == "done":
                self.downloader.cleanup(job_id)

    def _save_evaluation(self, candidate, plan, directory, verification, evaluation, job_id=None,
                         seed_id=None, rel_dir=None):
        seed_id = seed_id or candidate.id
        rel_dir = rel_dir or candidate.id
        plan.runtime_config.backend = evaluation.backend
        plan.runtime_config.runner_version = evaluation.runner_version
        if evaluation.backend == "cpu":
            plan.runtime_config.gpu_layers = 0
        offline_check = evaluation.detail.get("offline_check", {})
        manifest = build_manifest(
            seed_id=seed_id, candidate=candidate, plan=plan, verification=verification,
            runtime_config=plan.runtime_config, evaluation=evaluation,
            hardware=self.hardware(), offline_check=offline_check)
        write_usage(directory, usage_markdown(manifest=manifest, candidate=candidate,
                                              evaluation=evaluation, offline_check=offline_check))
        write_manifest(directory, manifest)
        reserve = self.inference.reserve(f"promote-{job_id}") if job_id else nullcontext()
        promote_files = self.downloader.promote(job_id, seed_id) if job_id else nullcontext(directory)
        with reserve, promote_files as seed_dir:
            ratio, missing = compute_completeness(seed_dir, manifest)
            gates_ok, reason = meets_gates(evaluation, load_eval_config())
            seed = Seed(id=seed_id, candidate_id=candidate.id, rel_dir=rel_dir, manifest=manifest,
                        completeness=ratio, offline_ready=gates_ok and not missing,
                        integrity="ok" if not missing else "incomplete")
            self._save_seed(seed, reason)
        return seed, reason

    def _save_seed(self, seed: Seed, reason: str) -> None:
        with self.db.transaction():
            self.db.put_seed(seed)
            self.db.set_candidate_state(seed.candidate_id,
                                        "offline_ready" if seed.offline_ready else "downloaded", reason)
            if not seed.offline_ready:
                for pointer in self.db.list_defaults():
                    if pointer.seed_id != seed.id:
                        continue
                    previous = self.db.get_seed(pointer.previous_seed_id) if pointer.previous_seed_id else None
                    if (previous and previous.id != seed.id and previous.offline_ready
                            and previous.integrity == "ok" and previous.manifest.get("task", "chat") == pointer.task):
                        pointer.seed_id = previous.id
                    else:
                        pointer.seed_id = None
                    pointer.previous_seed_id = seed.id
                    pointer.pinned = False
                    pointer.source = "auto"
                    self.db.set_default(pointer)

    def retest(self, seed_id: str) -> dict:
        with self.library_lock(wait=True):
            seed = self.db.get_seed(seed_id)
            if not seed:
                raise ValueError(f"种子不存在：{seed_id}")
            manifest = seed.manifest
            directory = self.paths.seeds / seed.rel_dir
            ok, missing, mismatched = verify_seed(directory, manifest)
            if not ok:
                seed.integrity = "incomplete"
                seed.offline_ready = False
                self._save_seed(seed, "归档完整性校验失败")
                return {"ok": False, "seed_id": seed_id, "errors": missing + mismatched}
            source = manifest.get("source", {})
            config = RuntimeConfig(**manifest.get("runtime_config", {}))
            if config.adapter not in {"llamacpp-openai", "mediapipe"}:
                raise ValueError("此归档尚未接入本机试用适配器")
            files = [FileSpec(repo_path=f["source_path"], local_path=f["relative_path"],
                              size_bytes=f["size_bytes"], kind=f["kind"], sha256=f["sha256"],
                              upstream_sha256=f.get("upstream_sha256")) for f in manifest["sizes_and_hashes"]]
            candidate = self.db.get_candidate(seed.candidate_id)
            if candidate is None:
                candidate = Candidate(id=seed.candidate_id or seed.id, source=source["name"],
                                      repo=source["repo"], commit_rev=source["fixed_revision"],
                                      variant=manifest["variant"], task=manifest["task"],
                                      license=source.get("license", ""), publisher=source.get("publisher", ""),
                                      files=files, size_bytes=sum(f.size_bytes for f in files))
                candidate = self.db.upsert_candidate(candidate)
                seed.candidate_id = candidate.id
            plan = DownloadPlan(candidate_id=candidate.id, repo=candidate.repo, commit_rev=source["fixed_revision"],
                                files=files, runtime_config=config)
            if config.adapter == "mediapipe":
                try:
                    weight = next(f.local_path for f in files if f.kind == "weight")
                    with self.inference.reserve(f"retest-{seed_id}"):
                        media.smoke(self.paths, candidate.task, directory / weight)
                except Exception as exc:
                    seed.offline_ready = False
                    reason = str(exc)
                else:
                    seed.offline_ready = True
                    reason = "本地进程空图检查通过"
                manifest["offline_verification"] = {"passed": seed.offline_ready, "method": reason}
                seed.integrity = "ok"
                write_manifest(directory, manifest)
                self._save_seed(seed, reason)
                return {"ok": seed.offline_ready, "seed_id": seed_id, "reason": reason}

            verification = {"files": [f.model_dump() for f in files],
                            "total_bytes": sum(f.size_bytes for f in files)}
            record = self.evaluator.evaluate(
                candidate=candidate, staging_dir=directory, plan=plan, config=config, verification=verification,
                real_drill=self.db.get_meta("real_drill") == "1")
            saved, reason = self._save_evaluation(candidate, plan, directory, verification, record,
                                                  seed_id=seed.id, rel_dir=seed.rel_dir)
            return {"ok": saved.offline_ready, "seed_id": seed_id, "reason": reason,
                    "evaluation": record.model_dump()}

    def _fail(self, job_id: str, state: str, message: str,
              candidate: Candidate | None) -> JobResult:
        self.db.update_job(job_id, state=state, stage=state, message=message[:1000])
        if candidate and state == "failed":
            # 失败不触碰现役种子与默认指针；保留已通过的种子与历史版本。
            # 已下载但未验证的包仍显示「已下载 · 待验证」，不与「离线可用」混淆。
            current = self.db.get_candidate(candidate.id)
            keep_state = "downloaded" if current and current.state == "downloaded" else "candidate"
            self.db.set_candidate_state(candidate.id, keep_state,
                                        f"本轮试装未完成：{message[:200]}")
        return JobResult(job_id, state, message)

    def _store_simple(self, job, candidate, plan, directory, verification, ready, reuse) -> JobResult:
        mode = "local" if ready else job.payload.get("mode", "requested")
        config = RuntimeConfig(adapter="mediapipe" if ready else "archive-only",
                               runner_version=media.VERSION if ready else "unregistered", verified=ready)
        check = {"passed": ready, "method": "本地进程空图检查" if ready else "仅校验收藏文件"}
        manifest = build_manifest(
            seed_id=candidate.id, candidate=candidate, plan=plan, verification=verification,
            runtime_config=config, evaluation=None, hardware=HardwareSample(), offline_check=check,
            external_references=[e.url for e in candidate.evidence if e.url],
        )
        manifest["collection_mode"] = mode
        manifest["runner"]["name"] = "MediaPipe" if ready else "未准备运行器"
        manifest["runner"]["command"] = []
        write_usage(directory, f"# {candidate.variant}\n\n"
                    + (f"通过 /api/infer 调用，task={candidate.task}，model={candidate.id}，inputs.image 传入图片的 Base64。"
                       if ready else "权重与附件已校验收藏，尚未进行本机试用验证。")
                    + "\n\n" + "\n".join(manifest["external_references"]) + "\n")
        write_manifest(directory, manifest)
        reserve = nullcontext() if reuse else self.inference.reserve(f"promote-{job.id}")
        promote_files = nullcontext() if reuse else self.downloader.promote(job.id, candidate.id)
        message = "离线可用" if ready else "已收藏 · 未在本机验证"
        with reserve, promote_files, self.db.transaction():
            self.db.put_seed(Seed(id=candidate.id, candidate_id=candidate.id, rel_dir=candidate.id,
                                  manifest=manifest, completeness=1.0, integrity="ok", offline_ready=ready))
            self.db.set_candidate_state(candidate.id, "offline_ready" if ready else "downloaded", message)
            self.db.update_job(job.id, state="done", stage="完成", progress=1.0, message=message,
                               payload={**job.payload, "seed_id": candidate.id})
        return JobResult(job.id, "done", message, candidate.id)

    def _update_defaults(self, seed_id: str, evaluation: EvaluationRecord,
                         assignments: list[dict[str, str]]) -> list[str]:
        """§4 换档门槛；保留旧版和用户固定的版本；恢复自动推荐时保留上个可用版本。"""
        config = load_eval_config()
        changed: list[str] = []
        for assignment in assignments:
            task, profile = assignment["task"], assignment["profile"]
            pointer = self.db.get_default(task, profile)
            if pointer.pinned:
                continue
            current: EvaluationRecord | None = None
            if pointer.seed_id:
                old_seed = self.db.get_seed(pointer.seed_id)
                if old_seed:
                    records = [e for e in self.db.list_evaluations(
                        candidate_id=old_seed.candidate_id, eval_version=config["version"])]
                    current = records[0] if records else None
                    if old_seed.offline_ready and old_seed.integrity == "ok" and current is None:
                        self.db.set_meta(f"promotion_skip:{task}:{profile}", "现任版本需按当前评测口径重测")
                        continue
            ok, reason = promote(current, evaluation, profile, config)
            if not ok:
                self.db.set_meta(f"promotion_skip:{task}:{profile}", reason)
                continue
            pointer.previous_seed_id = pointer.seed_id
            pointer.seed_id = seed_id
            pointer.pinned = False
            pointer.source = "auto"
            self.db.set_default(pointer)
            changed.append(f"{task}:{profile}（{reason}）")
        return changed
