"""每日发现（文档 §5 第 1 步）。

Hugging Face 优先用 HfApi.list_models / model_info，按订阅任务、可信发布者、
最近更新分页扫描；每个来源保留成功游标和少量重叠窗口，按仓库 + commit 去重；
单源出错保留旧结果；429 遵从平台限流规则，带退避和重试上限。
标签缺失的模型可从已知厂商官方发布入口或手动仓库地址补入。
"""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from .config import (DEFAULT_TRUSTED_PUBLISHERS, HF_BACKOFF_BASE_SECONDS,
                     HF_MAX_RETRIES, HF_MAX_WORKERS)
from .hardware import estimate_footprint
from .models import Candidate, Estimate, Evidence, FileSpec, Settings

LICENSE_ALLOWLIST = {
    "apache-2.0", "mit", "bsd-3-clause", "bsd-2-clause", "cc-by-4.0",
    "cc-by-sa-4.0", "cc0-1.0", "qwen", "qwen-research", "llama3", "gemma",
}
MODEL_CARD_NAMES = ("README.md", "readme.md", "modelcard.md")
LICENSE_NAMES = ("LICENSE", "LICENSE.txt", "LICENSE.md", "license")
USAGE_NAMES = ("USAGE.md", "usage.md", "docs/usage.md", "examples/README.md")
GGUF_SHARD = re.compile(r"(.+)-(\d{5})-of-(\d{5})\.gguf", re.I)
CONFIG_NAMES = ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json")

LIBRARY_SEARCH = {
    "chat": ("text-generation", ""), "code": ("text-generation", "coder"),
    "math": ("text-generation", "math"), "translation": ("translation", ""),
    "vision": ("image-text-to-text", ""), "ocr": ("image-text-to-text", "ocr"),
    "image_gen": ("text-to-image", ""), "image_edit": ("image-to-image", "edit"),
    "face_swap": ("image-to-image", "face-swap"), "speech": ("automatic-speech-recognition", ""),
    "tts": ("text-to-speech", ""), "video": ("video-classification", ""),
    "video_gen": ("text-to-video", ""), "retrieval": ("feature-extraction", ""),
}

LIBRARY_BENCHMARKS = {
    "chat": "TIGER-Lab/MMLU-Pro", "code": "SWE-bench/SWE-bench_Verified",
    "math": "openai/gsm8k", "vision": "MMMU/MMMU_Pro", "ocr": "allenai/olmOCR-bench",
    "speech": "hf-audio/open-asr-leaderboard", "retrieval": "mteb/arguana",
    "video": "MME-Benchmarks/Video-MME-v2",
}


@dataclass
class DiscoveryResult:
    candidates: list[Candidate] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)
    cursors: dict[str, str] = field(default_factory=dict)
    status: str = "ok"          # ok | partial | retained
    scanned: int = 0

    @property
    def retained_previous(self) -> bool:
        return self.status == "retained"


class HuggingFaceSource:
    name = "huggingface"

    @property
    def supports_public_reviews(self) -> bool:
        return self._api is None or hasattr(self._api, "dataset_info")

    def __init__(self, db, api: Any | None = None, sleep: Callable[[float], None] = time.sleep):
        self.db = db
        self._api = api
        self.sleep = sleep

    # ---------- 基础设施 ----------
    def api(self):
        if self._api is not None:
            return self._api
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:  # 离线启动不应因缺依赖崩溃
            raise RuntimeError("huggingface_hub 未安装，来源刷新不可用；已保留旧目录") from exc
        self._api = HfApi()
        return self._api

    def _call(self, fn: Callable[[], Any]) -> Any:
        last: Exception | None = None
        for attempt in range(HF_MAX_RETRIES):
            try:
                return fn()
            except Exception as exc:  # huggingface_hub / httpx 的异常类型各自不同
                last = exc
                response = getattr(exc, "response", None)
                status = getattr(response, "status_code", None)
                if status == 429 or "429" in str(exc) or "rate limit" in str(exc).lower():
                    self.sleep(HF_BACKOFF_BASE_SECONDS * (2 ** attempt))
                    continue
                if status and 500 <= int(status) < 600:
                    self.sleep(HF_BACKOFF_BASE_SECONDS * (2 ** attempt))
                    continue
                raise
        raise RuntimeError(f"达到重试上限（{HF_MAX_RETRIES} 次）：{last}")

    # ---------- 扫描 ----------
    def discover_library(self, settings: Settings) -> DiscoveryResult:
        """查找各类型的固定文件清单。人体和手部使用应用登记的官方模型。"""
        from .researcher import RESEARCH_REPOS

        result = DiscoveryResult()
        try:
            api = self.api()
        except Exception as exc:
            error = str(exc)
            self.db.mark_source_error(self.name, error)
            return DiscoveryResult(status="retained", errors=[{"source": self.name, "error": error}])

        def search(entry):
            task, (tag, query) = entry
            try:
                items = self._call(lambda: list(api.list_models(
                    pipeline_tag=tag, search=query or None, gated=False, sort="downloads",
                    limit=4, token=False)))
                return [item.id for item in items[:4]], None
            except Exception as exc:
                return [], {"source": self.name, "task": task, "error": str(exc)[:300]}

        pinned = {repo.casefold(): task for task, repo in RESEARCH_REPOS.items()}
        repos = dict.fromkeys([*RESEARCH_REPOS.values(), *settings.manual_repos])
        with ThreadPoolExecutor(max_workers=HF_MAX_WORKERS) as pool:
            for hits, error in pool.map(search, LIBRARY_SEARCH.items()):
                repos.update(dict.fromkeys(hits))
                if error:
                    result.errors.append(error)

        def load(repo):
            try:
                info = self._call(lambda: api.model_info(repo, files_metadata=True, token=False))
                models = self._to_candidates(info)
                for model in models:
                    model.task = pinned.get(model.repo.casefold(), model.task)
                return models, None
            except Exception as exc:
                return [], {"source": self.name, "repo": repo, "error": str(exc)[:300]}

        result.scanned = len(repos)
        candidates = {}
        with ThreadPoolExecutor(max_workers=HF_MAX_WORKERS) as pool:
            for models, error in pool.map(load, repos):
                candidates.update((c.id, c) for c in models)
                if error:
                    result.errors.append(error)
        leaderboard = getattr(api, "get_dataset_leaderboard", None)
        if leaderboard:
            def leader(entry):
                task, dataset = entry
                try:
                    rows = leaderboard(dataset, token=False, timeout=12)
                    for row in rows[:12]:
                        try:
                            info = api.model_info(row.model_id, files_metadata=True, token=False, timeout=12)
                        except Exception as exc:
                            if getattr(getattr(exc, "response", None), "status_code", None) in {401, 403, 404}:
                                continue
                            raise
                        original = next((c for c in self._to_candidates(info)
                                         if c.variant.endswith(" · 原始文件") and c.state != "gated"
                                         and c.license in LICENSE_ALLOWLIST and c.size_bytes > 0
                                         and all(f.size_bytes > 0 for f in c.files if f.kind == "weight")), None)
                        if original is None:
                            continue
                        original.task = task
                        original.evidence.append(Evidence(
                            title=dataset, url=f"https://huggingface.co/datasets/{dataset}/leaderboard",
                            source="Hugging Face 公开榜单", dataset=dataset,
                            model_version=original.commit_rev, variant=original.variant,
                            benchmark_key=f"hf-board:{dataset}", metric="榜单得分",
                            value=row.value, public_rank=row.rank,
                            note=("榜单关联发布模型，未绑定权重修订。"
                                  + ("榜单标记为已验证。" if row.verified else "公开报告成绩。")
                                  + (f" {row.notes[:240]}" if row.notes else "")),
                        ))
                        return task, original, None
                    return task, None, None
                except Exception as exc:
                    return task, None, {"source": self.name, "task": task, "error": str(exc)[:300]}

            with ThreadPoolExecutor(max_workers=HF_MAX_WORKERS) as pool:
                for task, winner, error in pool.map(leader, LIBRARY_BENCHMARKS.items()):
                    if error:
                        result.errors.append(error)
                        continue
                    board_key = f"hf-board:{LIBRARY_BENCHMARKS[task]}"
                    for old in self.db.list_candidates(task=task):
                        if any(e.benchmark_key == board_key for e in old.evidence):
                            old.evidence = [e for e in old.evidence if e.benchmark_key != board_key]
                            candidates.setdefault(old.id, old)
                    if winner:
                        candidates[winner.id] = winner
        result.candidates = list(candidates.values())
        if result.errors:
            result.status = "partial"
            self.db.mark_source_error(self.name, result.errors[0]["error"])
        else:
            self.db.mark_source_success(self.name, self.db.get_source(self.name).get("cursor"))
        return result

    def cache_model_cards(self, candidates: list[Candidate]) -> None:
        """缓存推荐版本的上游说明。只取非权重的小型文本。"""
        try:
            download = getattr(self.api(), "hf_hub_download", None)
        except Exception:
            return
        if download is None:
            return

        jobs = {}
        for candidate in candidates:
            if candidate.source != "huggingface":
                continue
            card = next((f for f in candidate.files if f.kind == "model_card"
                         and 0 < f.size_bytes <= 96 * 1024), None)
            if card is None:
                continue
            key = f"model-card:{candidate.repo}:{candidate.commit_rev}"
            if not self.db.get_meta(key):
                jobs[key] = (candidate, card.repo_path)

        def save(entry):
            key, (candidate, filename) = entry
            try:
                path = download(candidate.repo, filename, revision=candidate.commit_rev,
                                token=False, etag_timeout=8)
                with Path(path).open("rb") as handle:
                    raw = handle.read(96 * 1024 + 1)
                if len(raw) <= 96 * 1024:
                    self.db.set_meta(key, raw.decode("utf-8", errors="replace"))
            except Exception:
                # 固定版本的摘要在后续刷新中可以重试。
                return

        with ThreadPoolExecutor(max_workers=HF_MAX_WORKERS) as pool:
            list(pool.map(save, jobs.items()))

    def discover(self, settings: Settings) -> DiscoveryResult:
        result = DiscoveryResult()
        scan_started = datetime.now(timezone.utc)
        cursor = self.db.get_source(self.name).get("cursor")
        scope = json.dumps(sorted(settings.trusted_publishers or DEFAULT_TRUSTED_PUBLISHERS))
        if self.db.get_meta("hf-paginated-scope") != scope:
            cursor = None
        overlap_cutoff: datetime | None = None
        if cursor:
            try:
                # 少量重叠窗口，避免边界漏项；重复由「仓库 + commit」去重吸收。
                overlap_cutoff = datetime.fromisoformat(cursor) - timedelta(hours=2)
            except ValueError:
                overlap_cutoff = None
        publishers = settings.trusted_publishers or list(DEFAULT_TRUSTED_PUBLISHERS)
        try:
            api = self.api()
        except Exception as exc:
            self.db.mark_source_error(self.name, str(exc))
            result.errors.append({"source": self.name, "error": str(exc)})
            result.status = "retained"
            return result

        def modified_at(item):
            modified = getattr(item, "last_modified", None) or getattr(item, "lastModified", None)
            if isinstance(modified, str):
                try:
                    modified = datetime.fromisoformat(modified.replace("Z", "+00:00"))
                except ValueError:
                    return None
            if modified is not None and modified.tzinfo is None:
                modified = modified.replace(tzinfo=timezone.utc)
            return modified

        def list_publisher(publisher: str) -> tuple[list[Any], str]:
            def scan():
                items = []
                for item in api.list_models(author=publisher, sort="last_modified",
                                            expand=["lastModified"], token=False):
                    modified = modified_at(item)
                    if overlap_cutoff and modified and modified < overlap_cutoff:
                        break
                    items.append(item)
                return items

            try:
                return self._call(scan), ""
            except Exception as exc:
                return [], f"{type(exc).__name__}: {str(exc)[:500]}"

        # 发布者失败保留旧游标，仍处理本轮其他可达来源。
        with ThreadPoolExecutor(max_workers=HF_MAX_WORKERS) as pool:
            batches = list(pool.map(list_publisher, publishers))
        failures = [error for _, error in batches if error]
        if failures:
            message = "; ".join(failures[:3])
            self.db.mark_source_error(self.name, message)
            result.errors.extend({"source": self.name, "error": error} for error in failures)
            result.status = "partial"

        repos: list[Any] = [repo for batch, _ in batches for repo in batch]
        for repo in settings.manual_repos:  # 手动仓库地址补入
            if repo and repo not in {getattr(r, "id", "") for r in repos}:
                repos.append(type("ManualRepo", (), {"id": repo, "lastModified": None})())

        # 先在主线程做去重和游标过滤，保持请求数量可控。
        ordered: list[str] = []
        seen: set[str] = set()
        for item in repos:
            repo_id = getattr(item, "id", None) or getattr(item, "modelId", None)
            if not repo_id or repo_id in seen:
                continue
            seen.add(repo_id)
            ordered.append(repo_id)
        result.scanned = len(ordered)

        def load_repo(repo_id: str) -> tuple[str, list[Candidate], str]:
            try:
                info = self._call(lambda: api.model_info(repo_id, files_metadata=True, token=False))
                return repo_id, self._to_candidates(info), ""
            except Exception as exc:
                return repo_id, [], str(exc)[:500]

        # 元数据读取最多 5 路并发；单个仓库失败只标记 partial，不丢弃其他结果。
        with ThreadPoolExecutor(max_workers=HF_MAX_WORKERS) as pool:
            loaded = list(pool.map(load_repo, ordered))
        for repo_id, candidates, error in loaded:
            if error:
                result.errors.append({"source": self.name, "repo": repo_id, "error": error})
                result.status = "partial"
                continue
            result.candidates.extend(candidates)

        if result.errors:
            result.status = "partial" if result.candidates else "retained"
            self.db.mark_source_error(self.name, "; ".join(e["error"] for e in result.errors[:3]))
        else:
            new_cursor = scan_started.isoformat(timespec="seconds")
            result.cursors[self.name] = new_cursor
        return result

    def pin(self, candidate: Candidate, revision: str | None = None) -> Candidate:
        if candidate.source != "huggingface":
            return candidate
        ref = revision or candidate.commit_rev
        weights = [f.repo_path for f in candidate.files if f.kind == "weight"]
        split = GGUF_SHARD.fullmatch(weights[0]) if weights else None
        complete = not split or weights == [
            f"{split[1]}-{number:05d}-of-{split[3]}.gguf" for number in range(1, int(split[3]) + 1)]
        if re.fullmatch(r"[0-9a-f]{40}", ref) and candidate.commit_rev == ref and weights and complete and all(
                f.size_bytes > 0 or f.kind not in ("weight", "projection") for f in candidate.files):
            return candidate
        info = self._call(lambda: self.api().model_info(candidate.repo, revision=ref, files_metadata=True, token=False))
        variants = self._to_candidates(info, all_variants=True)
        fixed = next((c for c in variants if c.variant == candidate.variant and c.repo == candidate.repo), None)
        if fixed is None or any(f.size_bytes <= 0 for f in fixed.files if f.kind in ("weight", "projection")):
            raise ValueError("无法确认固定版本或完整权重大小")
        if re.fullmatch(r"[0-9a-f]{40}", ref) and fixed.commit_rev != ref:
            raise ValueError("上游返回的提交与请求不符")
        return self.db.upsert_candidate(fixed)

    # ---------- 转换 ----------
    @staticmethod
    def _siblings(info: Any) -> list[dict[str, Any]]:
        out = []
        for s in getattr(info, "siblings", []) or []:
            lfs = getattr(s, "lfs", None) or {}
            def field(name, default=None):
                return lfs.get(name, default) if isinstance(lfs, dict) else getattr(lfs, name, default)
            digest = field("sha256") or field("oid_sha256")
            out.append({
                "name": getattr(s, "rfilename", "") or "",
                "size": int(getattr(s, "size", 0) or field("size", 0) or 0),
                "sha256": digest if isinstance(digest, str) and len(digest) == 64 else None,
            })
        return out

    def _to_candidates(self, info: Any, *, all_variants=False) -> list[Candidate]:
        repo = getattr(info, "id", "") or ""
        commit = getattr(info, "sha", "") or ""
        if not repo or not re.fullmatch(r"[0-9a-f]{40}", commit):
            return []
        card = getattr(info, "cardData", None) or {}
        if not hasattr(card, "get"):
            card = {}
        license_id = str(card.get("license", "") or "").lower()
        publisher = repo.split("/")[0] if "/" in repo else ""
        pipeline = str(getattr(info, "pipeline_tag", None) or card.get("pipeline_tag", "") or "")
        task = _pipeline_to_task(pipeline, repo)
        gated = str(getattr(info, "gated", "") or "")
        siblings = self._siblings(info)
        by_name = {s["name"]: s for s in siblings}
        groups = {}
        for sibling in siblings:
            name = sibling["name"]
            if not name.lower().endswith(".gguf") or "mmproj" in name.lower():
                continue
            split = GGUF_SHARD.fullmatch(name)
            key = (split[1], int(split[3])) if split else (name, 1)
            groups.setdefault(key, []).append((int(split[2]) if split else 1, sibling))
        weights = []
        for (_, count), parts in groups.items():
            parts.sort(key=lambda part: part[0])
            if len(parts) == count and all(number == i for i, (number, _) in enumerate(parts, 1)):
                weights.append([item for _, item in parts])
        weights.sort(key=lambda group: (0 if "Q4_K_M" in group[0]["name"] else
                                        1 if "Q4_0" in group[0]["name"] else 2,
                                        sum(s["size"] for s in group)))
        if not all_variants:
            weights = weights[:2]
        package = [s for s in siblings if s["name"].lower().endswith(
            (".safetensors", ".onnx", ".onnx_data", ".bin", ".pth"))]

        evidence = [Evidence(
            title="上游模型卡与文件清单",
            url=f"https://huggingface.co/{repo}/tree/{commit}",
            source="Hugging Face 模型卡", model_version=commit,
            dataset="", protocol="",
            note="原始精度成绩只作同家族线索；具体量化版标「待测」",
        )]
        for block in (card.get("model-index") or [])[:2]:
            if not isinstance(block, dict):
                continue
            for res in (block.get("results") or [])[:2]:
                if not isinstance(res, dict):
                    continue
                evidence.append(Evidence(
                    title=str((res.get("task") or {}).get("type", "公开评测")),
                    source="模型卡 model-index", model_version=commit,
                    dataset=str((res.get("dataset") or {}).get("name", "")),
                    protocol=str(res.get("metrics", "")),
                    note="公开证据，不等于本机实测",
                ))

        out: list[Candidate] = []
        for group in weights:
            variant = group[0]["name"].rsplit("/", 1)[-1]
            weight_bytes = sum(part["size"] for part in group)
            files = [FileSpec(repo_path=part["name"], local_path=part["name"].rsplit("/", 1)[-1],
                              size_bytes=part["size"], kind="weight", upstream_sha256=part["sha256"])
                     for part in group]
            attachments = 0
            for group, kind in ((MODEL_CARD_NAMES, "model_card"), (LICENSE_NAMES, "license"),
                                (USAGE_NAMES, "usage"), (CONFIG_NAMES, "config")):
                for name in group:
                    if name in by_name and name not in {f.repo_path for f in files}:
                        f = by_name[name]
                        files.append(FileSpec(repo_path=name, local_path="upstream/" + name,
                                              size_bytes=f["size"], kind=kind,
                                              upstream_sha256=f["sha256"]))
                        attachments += f["size"]
            # 图文模型带齐匹配的投影文件。
            for s in siblings:
                if "mmproj" in s["name"].lower() and s["name"].endswith(".gguf"):
                    files.append(FileSpec(repo_path=s["name"],
                                          local_path=s["name"].split("/")[-1],
                                          size_bytes=s["size"], kind="projection",
                                          upstream_sha256=s["sha256"]))
                    attachments += s["size"]
            total = weight_bytes + attachments
            projection = sum(f.size_bytes for f in files if f.kind == "projection")
            backend = "cuda" if weight_bytes > 1_500_000_000 else "cpu"
            footprint = estimate_footprint(weight_bytes, projection_bytes=projection,
                                           backend=backend, gpu_layers=99 if backend == "cuda" else 0)
            state, reason = "candidate", ""
            if gated and gated.lower() != "false":
                state, reason = "gated", "受限访问模型：需用户在线办理授权"
            elif not license_id:
                reason = "许可未确认，停在候选状态"
            elif license_id not in LICENSE_ALLOWLIST:
                state, reason = "candidate", f"许可 {license_id} 不在首版白名单，停在候选状态"
            elif not TASK_ENABLED.get(task, False):
                reason = f"{task} 属于后续阶段，本轮只登记元数据"
            out.append(Candidate(
                id=Candidate.make_id("huggingface", repo, commit, variant),
                source="huggingface", repo=repo, commit_rev=commit, variant=variant,
                task=task, publisher=publisher, license=license_id, size_bytes=total,
                files=files, evidence=evidence,
                estimate=Estimate(backend=footprint["backend"],
                                  est_ram_mib=footprint["est_ram_mib"],
                                  est_vram_mib=footprint["est_vram_mib"],
                                  basis=footprint["basis"]),
                rationale="按订阅任务、可信发布者、最近更新分页扫描得到的候选；兼容性与档位待本机实测",
                state=state, reason=reason,
            ))
        if package:
            files = [FileSpec(repo_path=s["name"], local_path="payload/" + s["name"], size_bytes=s["size"],
                              kind="weight", upstream_sha256=s["sha256"]) for s in package]
            for s in siblings:
                name = s["name"]
                if name.lower().endswith((".json", ".md", ".txt", ".model", ".vocab", ".tiktoken")) or name in LICENSE_NAMES:
                    files.append(FileSpec(
                        repo_path=name, local_path="payload/" + name, size_bytes=s["size"], upstream_sha256=s["sha256"],
                        kind="license" if name in LICENSE_NAMES else "model_card" if name in MODEL_CARD_NAMES else "config",
                    ))
            variant = repo.rsplit("/", 1)[-1] + " · 原始文件"
            out.append(Candidate(
                id=Candidate.make_id("huggingface", repo, commit, variant),
                repo=repo, commit_rev=commit, variant=variant, task=task,
                publisher=publisher, license=license_id, files=files,
                size_bytes=sum(f.size_bytes for f in files),
                evidence=evidence + _exact_scores(card, repo, commit, variant),
                state="gated" if gated and gated.lower() != "false" else "candidate",
                rationale="原始权重与配套文件可用于收藏，试用取决于已接入的适配器。",
            ))
        return out


def _exact_scores(card: dict, repo: str, revision: str, variant: str) -> list[Evidence]:
    scores = []
    for block in card.get("model-index") or []:
        if not isinstance(block, dict) or block.get("name") not in (repo, repo.rsplit("/", 1)[-1]):
            continue
        for result in block.get("results") or []:
            if not isinstance(result, dict) or result.get("variant") != variant or result.get("revision") != revision:
                continue
            dataset = result.get("dataset") or {}
            source = result.get("source") or {}
            if not all(dataset.get(k) for k in ("type", "revision", "split")) or not source.get("url") or not result.get("protocol"):
                continue
            for metric in result.get("metrics") or []:
                kind = str(metric.get("type", "")).lower()
                value = metric.get("value")
                unit = metric.get("unit")
                if (kind not in {"accuracy", "acc", "f1", "map", "iou", "wer", "cer", "bleu"}
                        or not isinstance(value, (float, int)) or isinstance(value, bool) or not unit):
                    continue
                key = json.dumps([dataset.get(k) for k in ("type", "revision", "config", "split", "args")] +
                                 [result.get("task"), source["url"], result["protocol"], metric.get("args"), unit],
                                 sort_keys=True, ensure_ascii=False, default=str)
                scores.append(Evidence(
                    title=str(dataset.get("name") or dataset["type"]), source=source.get("name") or "上游自报",
                    url=f"https://huggingface.co/{repo}/blob/{revision}/README.md", model_version=revision,
                    dataset=str(dataset["type"]), protocol=result["protocol"],
                    benchmark_key="explicit-v2:" + key, variant=variant, metric=kind, value=value,
                    higher_is_better=kind not in {"wer", "cer"},
                ))
    return scores


def _pipeline_to_task(pipeline: str, repo: str = "") -> str:
    name = repo.rsplit("/", 1)[-1].lower()
    if re.search(r"face[-_]?swap|inswapper", name):
        return "face_swap"
    if re.search(r"(?:^|[-_.])ocr(?:$|[-_.])|paddleocr", name):
        return "ocr"
    if pipeline == "text-generation":
        if re.search(r"(?:^|[-_.])(?:coder|code)(?:$|[-_.])", name):
            return "code"
        if re.search(r"(?:^|[-_.])math(?:$|[-_.])", name):
            return "math"
    return {
        "translation": "translation", "sentence-similarity": "retrieval",
        "document-question-answering": "ocr", "video-text-to-text": "video",
        "text-generation": "chat", "image-text-to-text": "vision",
        "automatic-speech-recognition": "speech", "text-to-image": "image_gen",
        "text-to-speech": "tts", "feature-extraction": "retrieval",
        "image-to-text": "vision", "image-to-image": "image_edit", "text-to-video": "video_gen",
        "video-to-text": "video", "video-classification": "video",
    }.get(pipeline, "chat")


# 阶段 1 只有通用对话接入下载链路（文档 §1、§8）。
TASK_ENABLED = {"chat": True, "pose": True, "hand": True}


def local_adapter(candidate: Candidate) -> str | None:
    if candidate.source == "huggingface" and candidate.task == "chat" and any(f.kind == "weight" and f.local_path.lower().endswith(".gguf")
                                        for f in candidate.files):
        return "llamacpp-openai"
    if candidate.source == "mediapipe" and candidate.task in {"pose", "hand"}:
        return "mediapipe"
    return None
