"""作业状态机与同步编排（文档 §5、§8）。

目录与作业分开记录。作业走 排队 → 下载 → 校验 → 调试 → 复测 → 完成，
异常进入「等待网络」「等待资源」或「失败」。完整度按实际文件与字节数计算。
重启后从 SQLite 持久化作业恢复；`来源 + 仓库 + commit + 变体` 用作下载幂等键。
默认版本是「能力 + 档位」的单独映射，只在复测与断网验收都通过后按档位规则切换。
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .agent import AgentGateway
from .archive import (build_manifest, compute_completeness, usage_markdown,
                      write_manifest, write_usage)
from .catalog import load_eval_config, meets_gates, promote, sota_leaders
from .config import Paths
from .db import Database
from .debug_tools import DebugToolbox
from .discovery import DiscoveryResult, HuggingFaceSource, LICENSE_ALLOWLIST, local_adapter
from .downloader import DownloadError, DownloadPlan, Downloader, resolve_fixed_commit
from .evaluation import Evaluator
from .hardware import probe as probe_hardware
from .models import (PROFILES, Candidate, EvaluationRecord, HardwareSample, Job, RuntimeConfig,
                     Seed, Settings, SyncReport, utc_now)
from .preselect import PreSelector, ShortlistResult
from .runner import InferenceService, RuntimeUnavailable
from .runtime_registry import RuntimeRegistry
from . import media


class SyncBusy(Exception):
    pass


class SyncLock:
    """应用层同一把同步锁：手动触发与计划任务冲突时明确跳过。"""

    def __init__(self, paths: Paths, db: Database, stale_seconds: float = 7200):
        self.path = paths.sync_lock
        self.db = db
        self.stale_seconds = stale_seconds
        self._thread_lock = threading.Lock()

    @contextmanager
    def hold(self, trigger: str = "manual"):
        acquired = self._thread_lock.acquire(blocking=False)
        if not acquired:
            raise SyncBusy("已有同步正在运行，同一计划忽略重复实例")
        try:
            if self.path.exists():
                try:
                    data = json.loads(self.path.read_text(encoding="utf-8"))
                    age = time.time() - float(data.get("ts", 0))
                    if age < self.stale_seconds:
                        raise SyncBusy(f"同步锁被占用（{data.get('trigger', 'unknown')}，{age:.0f} 秒前）")
                except (ValueError, TypeError):
                    pass
                self.path.unlink(missing_ok=True)
            self.path.write_text(json.dumps({"ts": time.time(), "pid": os.getpid(),
                                             "trigger": trigger}), encoding="utf-8")
            try:
                yield
            finally:
                self.path.unlink(missing_ok=True)
        finally:
            self._thread_lock.release()


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
        self.sync_lock = SyncLock(paths, db)
        self._workers: list[threading.Thread] = []
        self._last_report: SyncReport | None = None

    def hardware(self) -> HardwareSample:
        return probe_hardware(self.paths, self.registry)

    # ---------- 同步 ----------
    def sync(self, trigger: str = "manual", wait: bool = False, catalog_only: bool = False) -> SyncReport:
        report = SyncReport(id=uuid.uuid4().hex[:16], trigger=trigger)
        self.db.start_sync(report)
        try:
            with self.sync_lock.hold(trigger):
                settings = self.db.get_settings()
                if settings.simulated_offline:
                    report.status = "skipped"
                    report.message = "模拟模式已暂停同步"
                    report.skipped.append({"reason": report.message})
                    self.db.set_meta("last_sync_attempt", utc_now())
                    self.db.set_meta("last_sync_skip_reason", report.message)
                    return report
                if not settings.auto_update and trigger == "scheduled":
                    report.status = "skipped"
                    report.message = "自动更新已关闭"
                    report.skipped.append({"reason": "自动更新已关闭"})
                    return report
                self.db.set_meta("last_sync_attempt", utc_now())
                result = self.source.discover(settings)
                result.candidates.extend(media.candidates())
                if catalog_only:
                    self.preselector.store_discovery(result, settings)
                    report.discovered = len(result.candidates)
                    report.status = "partial" if result.errors else "ok"
                    self.db.set_meta("last_sync_attempt", utc_now())
                    if not result.errors:
                        self.db.set_meta("last_sync_success", utc_now())
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
                if not job_ids:
                    self.db.set_meta("last_sync_success", utc_now())
                self._start_worker(job_ids, wait=wait, report=report)
        except SyncBusy as exc:
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
        target = lambda: self._run_jobs(job_ids, report)
        if wait:
            target()
        else:
            thread = threading.Thread(target=target, daemon=True, name="doomsday-jobs")
            self._workers.append(thread)
            thread.start()

    def _run_jobs(self, job_ids: list[str], report: SyncReport | None = None) -> None:
        # 单次只加载一个模型：作业串行执行，后台评测由资源锁排队。
        for job_id in job_ids:
            self.run_job(job_id)
        if report:
            self.db.set_meta("last_sync_success", utc_now())

    def recover(self) -> list[str]:
        jobs = self.db.recover_jobs()
        return [j.id for j in jobs]

    def run_pending(self) -> None:
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
        adapter = local_adapter(candidate)
        if mode == "local" and not adapter:
            raise ValueError("此版本暂不支持本机试用")
        if mode == "sota" and candidate.id not in sota_leaders(self.db.list_candidates(task=task)):
            raise ValueError("此版本目前没有可用于 SOTA 收藏的领先记录")
        config = RuntimeConfig(adapter=adapter or "archive-only")
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
        job = self.db.get_job(job_id)
        if job is None or job.state != "queued":
            return JobResult(job_id, job.state if job else "missing", "作业无需执行")
        claim = self.db.execute("UPDATE jobs SET state='downloading', stage='下载', updated_at=? "
                                "WHERE id=? AND state='queued'", (utc_now(), job_id))
        if not claim.rowcount:
            return JobResult(job_id, "downloading", "作业已由其他进程接手")
        if job.kind == "local_import":
            from .local_import import run
            return run(self, job)
        plan = DownloadPlan(**job.payload["plan"])
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
            if candidate.source != "mediapipe" and not re.fullmatch(r"[0-9a-f]{7,40}", plan.commit_rev or ""):
                plan.commit_rev = resolve_fixed_commit(plan.repo, plan.commit_rev)
            archived = self.db.find_seed_by_candidate(candidate.id)
            reuse = archived and archived.integrity == "ok" and archived.manifest["source"]["fixed_revision"] == plan.commit_rev
            preflight = self.downloader.preflight(plan, job_id)
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
                    media.smoke(self.paths, candidate.task, files_dir / candidate.variant)
                return self._store_simple(job, candidate, plan, files_dir, verification, ready, bool(reuse))

            # 调试（Agent 只操作暂存区与回环服务）
            self.db.update_job(job_id, state="debugging", stage="调试", progress=0.72,
                               message="正在准备本地对话环境")
            if plan.runtime_config.runner_version == "unregistered":
                try:
                    self.registry.prepare_cpu()
                except Exception as exc:
                    raise DownloadError("对话环境暂不可用，已保留权重", "network") from exc
                current = self.hardware()
                self.db.save_sample(current)
                backend = "cuda" if current.backend_available.get("cuda") else "cpu"
                plan.runtime_config = RuntimeConfig(backend=backend, runner_version=self.registry.version(),
                                                    gpu_layers=99 if backend == "cuda" else 0)
            tuned, diagnostics = self.debug_box.run_debug(
                job_id, plan, plan.runtime_config, self.gateway)
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
            offline_check = evaluation.detail.get("offline_check", {})
            manifest = build_manifest(
                seed_id=candidate.id, candidate=candidate, plan=plan,
                verification=verification, runtime_config=tuned, evaluation=evaluation,
                hardware=self.hardware(), offline_check=offline_check)
            write_manifest(files_dir, manifest)
            write_usage(files_dir, usage_markdown(manifest=manifest, candidate=candidate,
                                                  evaluation=evaluation,
                                                  offline_check=offline_check))
            seed_dir = files_dir if reuse else self.downloader.promote(job_id, candidate.id)
            completeness, missing = compute_completeness(seed_dir, manifest)
            gates_ok, gates_reason = meets_gates(evaluation, load_eval_config())
            offline_ready = bool(evaluation.offline_ok and gates_ok and not missing)
            seed = Seed(id=candidate.id, candidate_id=candidate.id,
                        rel_dir=candidate.id, manifest=manifest,
                        completeness=completeness, offline_ready=offline_ready,
                        integrity="ok" if not missing else "incomplete")

            with self.db.transaction() as _conn:
                self.db.put_seed(seed)
                if offline_ready:
                    self.db.set_candidate_state(candidate.id, "offline_ready",
                                                "断网启动与任务样例通过，离线可用")
                else:
                    self.db.set_candidate_state(
                        candidate.id, "downloaded",
                        f"文件已下载，正在验证离线启动。（{gates_reason}）")
            promoted = self._update_defaults(candidate.id, evaluation, assignments) if offline_ready else []
            message = ("离线可用，已归档" if offline_ready else
                       "复测未过线，已归档为「已下载 · 待验证」，默认指针不变")
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
        mode = "local" if ready else "sota"
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
        if not reuse:
            self.downloader.promote(job.id, candidate.id)
        message = "离线可用" if ready else "已收藏 · 未在本机验证"
        with self.db.transaction():
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
                        candidate_id=old_seed.candidate_id, eval_version=config["version"])
                        if e.hardware_fingerprint == evaluation.hardware_fingerprint]
                    current = records[0] if records else None
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
