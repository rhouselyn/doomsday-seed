"""预选与限量下载（文档 §5 第 2、3 步）。

脚本先按许可、架构、后端和资源预算筛掉明显不适用的变体；Agent 再给结构化建议。
每轮每项能力、每个订阅档位最多自动试装一个新变体；同一变体入围多档时只下载一次，
已归档版本直接复用。受限模型留在「待授权」，无法确认来源或许可时停在候选状态。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .agent import AgentContext, AgentGateway
from .catalog import CONTROLLED_CANDIDATES, load_eval_config
from .config import DEFAULT_TRUSTED_PUBLISHERS, FIXED_CONCURRENCY, FIXED_CONTEXT, MAX_NEW_VARIANTS_PER_TASK_PROFILE_PER_RUN, Paths
from .db import Database
from .discovery import DiscoveryResult, HuggingFaceSource, LICENSE_ALLOWLIST, local_adapter
from .hardware import budget, check_capacity, estimate_footprint
from .models import Candidate, DownloadPlan, Estimate, Evidence, FileSpec, HardwareSample, RuntimeConfig, Settings


def controlled_candidates() -> list[Candidate]:
    """把文档 §3 的启动候选登记为目录条目（尚未实测，不认定档位冠军）。"""
    out: list[Candidate] = []
    for c in CONTROLLED_CANDIDATES:
        out.append(Candidate(
            id=Candidate.make_id(c["source"], c["repo"], c["commit_rev"], c["variant"]),
            source=c["source"], repo=c["repo"], commit_rev=c["commit_rev"],
            variant=c["variant"], task=c["task"], publisher=c["publisher"],
            license=c["license"], size_bytes=c["size_bytes"],
            files=[FileSpec(repo_path=name, local_path=name.split("/")[-1],
                            kind="weight" if name.endswith(".gguf") else
                            "license" if "LICENSE" in name.upper() else "model_card")
                   for name in c["files"]],
            evidence=[Evidence(**e) for e in c.get("evidence", [])],
            estimate=Estimate(backend=c["backend"], basis="启动候选；文件元数据待下载预检补齐"),
            rationale="[受控候选] " + c["rationale"],
        ))
    return out


@dataclass
class ShortlistResult:
    plans: list[DownloadPlan] = field(default_factory=list)
    assignments: list[dict[str, str]] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)
    agent_note: str = ""
    controlled: bool = False
    suggestions: list[dict[str, Any]] = field(default_factory=list)


class PreSelector:
    def __init__(self, db: Database, paths: Paths, gateway: AgentGateway | None = None,
                 commit_resolver: Callable[[str, str], str] | None = None,
                 source: HuggingFaceSource | None = None):
        self.db = db
        self.paths = paths
        self.gateway = gateway or AgentGateway()
        # 可注入以便离线测试；生产环境把分支引用解析成不可变 commit。
        self.commit_resolver = commit_resolver
        self.source = source or HuggingFaceSource(db)

    def store_discovery(self, result: DiscoveryResult, settings: Settings) -> list[Candidate]:
        """目录与作业分开记录；受控候选始终在目录中，供 Agent 失败时兜底。"""
        for candidate in controlled_candidates():
            self.db.upsert_candidate(candidate)
        stored: list[Candidate] = []
        with self.db.transaction():
            for candidate in result.candidates:
                stored.append(self.db.upsert_candidate(candidate))
            for source, cursor in result.cursors.items():
                self.db.mark_source_success(source, cursor)
            if "huggingface" in result.cursors:
                scope = sorted(settings.trusted_publishers or DEFAULT_TRUSTED_PUBLISHERS)
                self.db.set_meta("hf-paginated-scope", json.dumps(scope))
        return stored

    def rule_filter(self, candidate: Candidate, sample: HardwareSample,
                    settings: Settings, *, installed=None, active=None) -> tuple[bool, str, dict[str, Any]]:
        if candidate.state == "gated":
            return False, "受限访问模型，留在「待授权」，由用户在线办理", {}
        if local_adapter(candidate) != "llamacpp-openai":
            return False, f"{candidate.task} 属于后续阶段，本轮只登记元数据", {}
        if not candidate.license or candidate.license not in LICENSE_ALLOWLIST:
            return False, "无法确认来源或许可，停在候选状态", {}
        if installed is None:
            installed = {seed.candidate_id for seed in self.db.list_seeds()}
        if candidate.id in installed:
            return False, "已归档版本直接复用，不重复下载", {}
        # 已存在未完成作业也由 create_job 的幂等键吸收，这里显式标注原因。
        if active is None:
            active = {r["candidate_id"] for r in self.db.query(
                "SELECT candidate_id FROM jobs WHERE state NOT IN ('done','failed')")}
        if candidate.id in active:
            return False, "同一变体已有在途作业，跨档复用", {}

        backend = candidate.estimate.backend or "cpu"
        if backend == "cuda" and not sample.backend_available.get("cuda", False):
            cpu_ok = sample.backend_available.get("cpu", False)
            if not cpu_ok:
                return False, "运行器未登记或后端不可用，作业标「待适配」", {}
            backend = "cpu"
        elif not sample.backend_available.get(backend, False):
            return False, f"{backend} 运行器未登记，等待安装锁定构建", {}

        capacity = {"fits": True, "reason": "文件元数据待下载预检补齐", "backend": backend}
        if candidate.size_bytes > 0:
            estimate = estimate_footprint(
                sum(f.size_bytes for f in candidate.files if f.kind == "weight"),
                projection_bytes=sum(f.size_bytes for f in candidate.files if f.kind == "projection"),
                backend=backend, gpu_layers=99 if backend == "cuda" else 0)
            capacity = check_capacity(
                estimate, sample, backend)
            if not capacity["fits"]:
                # §3「允许建议更低配置或已验证的 CPU 版本；没有合适版本就显示本机暂无可运行模型」
                suggestion = ("显存余量不足。可试用速度优先档，或稍后重试。"
                              if backend == "cuda" else "内存余量不足，请稍后重试。")
                return False, f"{capacity['reason']}；{suggestion}", capacity
        return True, "", {**capacity, "backend": backend}

    def build(self, result: DiscoveryResult, sample: HardwareSample,
              settings: Settings, registry_version: str = "unregistered") -> ShortlistResult:
        out = ShortlistResult()
        self.store_discovery(result, settings)
        if not any(task == "chat" for task, _ in settings.active_pairs()):
            return out
        candidates = self.db.list_candidates()
        # 受控候选是目录兜底；若同一变体已有固定 commit 的真实候选，排除分支引用版本，
        # 避免同一变体被下载两次（§5 跨档/跨来源去重）。
        fixed_pairs = {(c.repo, c.variant) for c in candidates
                       if re.fullmatch(r"[0-9a-f]{7,40}", c.commit_rev or "")}

        installed = {seed.candidate_id for seed in self.db.list_seeds()}
        active = {r["candidate_id"] for r in self.db.query(
            "SELECT candidate_id FROM jobs WHERE state NOT IN ('done','failed')")}
        eligible: list[dict[str, Any]] = []
        by_key: dict[str, Candidate] = {}
        seen_rv: set[tuple[str, str]] = set()
        for candidate in candidates:
            ok, reason, capacity = self.rule_filter(candidate, sample, settings, installed=installed, active=active)
            by_key[candidate.idempotency_key] = candidate
            controlled = candidate.rationale.startswith("[受控候选]")
            if controlled and (candidate.repo, candidate.variant) in fixed_pairs:
                out.skipped.append({"candidate_id": candidate.id, "repo": candidate.repo,
                                    "variant": candidate.variant,
                                    "reason": "同一变体已有固定 commit 候选，受控占位条目不重复试装"})
                continue
            rv = (candidate.repo, candidate.variant)
            if rv in seen_rv:
                continue
            seen_rv.add(rv)
            if not ok:
                out.skipped.append({"candidate_id": candidate.id,
                                    "repo": candidate.repo, "variant": candidate.variant,
                                    "reason": reason})
                if "运行器未登记" in reason:
                    self.db.set_candidate_state(candidate.id, "unsupported", reason)
                continue
            controlled = candidate.rationale.startswith("[受控候选]")
            eligible.append({
                "id": candidate.id, "repo": candidate.repo, "commit_rev": candidate.commit_rev,
                "variant": candidate.variant, "task": candidate.task,
                "license": candidate.license, "size_bytes": candidate.size_bytes,
                "files": [f.repo_path for f in candidate.files],
                "backend": capacity["backend"], "eligible": True, "controlled": controlled,
                "evidence_count": len(candidate.evidence),
                "evidence": [e.model_dump() for e in candidate.evidence],
                "est_ram_mib": candidate.estimate.est_ram_mib,
                "est_vram_mib": candidate.estimate.est_vram_mib,
                "estimate_basis": candidate.estimate.basis,
            })

        existing = [
            {"candidate_id": e.candidate_id, "quality": e.quality,
             "median_latency_ms": e.median_latency_ms, "backend": e.backend,
             "eval_version": e.eval_version}
            for e in self.db.list_evaluations(eval_version=load_eval_config()["version"])[:20]
        ]
        pairs = settings.active_pairs()
        resources = budget(sample, "cuda" if sample.backend_available.get("cuda") else "cpu")
        context = AgentContext(
            candidates=eligible, budget={"ram_usable_mib": resources.ram_usable_mib,
                                         "vram_usable_mib": resources.vram_usable_mib},
            hardware=sample.model_dump(), existing_evaluations=existing,
            subscriptions=[list(p) for p in pairs],
            max_suggestions=max(1, min(len(pairs), 30)),
        )
        if not eligible:
            context.max_suggestions = 1
        suggestions, out.controlled, out.agent_note = self.gateway.recommend(context)
        # 受控候选只在 Agent 连接失败时由 AgentGateway 返回；无建议 = 本轮只更新目录。
        out.suggestions = [s.model_dump() for s in suggestions]

        # 限量：每个 (能力, 档位) 最多一个新变体；跨档同变体只建一个作业。
        seen: set[str] = set()
        for task, profile in pairs:
            added = 0
            for suggestion in suggestions:
                if added >= MAX_NEW_VARIANTS_PER_TASK_PROFILE_PER_RUN:
                    break
                if suggestion.task != task or (suggestion.profile and suggestion.profile != profile):
                    continue
                key = f"huggingface|{suggestion.repo}|{suggestion.commit_rev}|{suggestion.variant}"
                candidate = by_key.get(key) or self.db.find_candidate_by_key(key)
                if candidate is None:
                    out.skipped.append({"repo": suggestion.repo, "variant": suggestion.variant,
                                        "reason": "建议未对应到目录中的候选，已忽略"})
                    continue
                try:
                    ref = candidate.commit_rev
                    if self.commit_resolver and not re.fullmatch(r"[0-9a-f]{40}", ref):
                        ref = self.commit_resolver(candidate.repo, ref)
                    fixed = self.source.pin(candidate, ref)
                    by_key[key] = fixed
                    candidate = fixed
                except Exception as exc:
                    out.skipped.append({"repo": suggestion.repo, "variant": suggestion.variant,
                                        "reason": f"无法确认固定版本的文件清单：{exc}"})
                    continue
                if candidate.id in seen:
                    out.assignments.append({"task": task, "profile": profile,
                                            "candidate_id": candidate.id,
                                            "reason": "同一变体入围多档，只下载一次"})
                    added += 1
                    continue
                ok, reason, capacity = self.rule_filter(candidate, sample, settings, installed=installed, active=active)
                if not ok:
                    out.skipped.append({"candidate_id": candidate.id, "repo": candidate.repo,
                                        "variant": candidate.variant, "reason": reason})
                    continue
                backend = capacity["backend"]
                runtime = RuntimeConfig(
                    adapter="llamacpp-openai", backend=backend,
                    runner_version=registry_version,
                    gpu_layers=99 if backend == "cuda" else 0,
                    context=FIXED_CONTEXT, concurrency=FIXED_CONCURRENCY,
                    mmproj=next((f.local_path for f in candidate.files
                                 if f.kind == "projection"), None),
                )
                plan = DownloadPlan(candidate_id=candidate.id, repo=candidate.repo,
                                    commit_rev=candidate.commit_rev, files=candidate.files,
                                    runtime_config=runtime, total_bytes=candidate.size_bytes)
                out.plans.append(plan)
                out.assignments.append({"task": task, "profile": profile,
                                        "candidate_id": candidate.id,
                                        "reason": suggestion.rationale})
                seen.add(candidate.id)
                added += 1
        return out
