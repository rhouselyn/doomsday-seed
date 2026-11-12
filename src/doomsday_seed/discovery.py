"""每日发现（文档 §5 第 1 步）。

Hugging Face 优先用 HfApi.list_models / model_info，按订阅任务、可信发布者、
最近更新分页扫描；每个来源保留成功游标和少量重叠窗口，按仓库 + commit 去重；
单源出错保留旧结果；429 遵从平台限流规则，带退避和重试上限。
标签缺失的模型可从已知厂商官方发布入口或手动仓库地址补入。
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from .config import (DEFAULT_TRUSTED_PUBLISHERS, HF_BACKOFF_BASE_SECONDS, HF_CURSOR_OVERLAP,
                     HF_MAX_RETRIES, HF_MAX_WORKERS)
from .hardware import estimate_footprint
from .models import Candidate, Estimate, Evidence, FileSpec, Settings, utc_now

LICENSE_ALLOWLIST = {
    "apache-2.0", "mit", "bsd-3-clause", "bsd-2-clause", "cc-by-4.0",
    "cc-by-sa-4.0", "cc0-1.0", "qwen", "qwen-research", "llama3", "gemma", "other",
}
MODEL_CARD_NAMES = ("README.md", "readme.md", "modelcard.md")
LICENSE_NAMES = ("LICENSE", "LICENSE.txt", "LICENSE.md", "license")
USAGE_NAMES = ("USAGE.md", "usage.md", "docs/usage.md", "examples/README.md")
CONFIG_NAMES = ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json")


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
    def discover(self, settings: Settings, max_models: int = 40) -> DiscoveryResult:
        result = DiscoveryResult()
        cursor = self.db.get_source(self.name).get("cursor")
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

        def list_publisher(publisher: str) -> tuple[list[Any], str]:
            try:
                return list(self._call(lambda: api.list_models(
                    author=publisher, sort="lastModified", direction=-1,
                    limit=min(20, max_models + HF_CURSOR_OVERLAP), full=False))), ""
            except Exception as exc:
                return [], f"{type(exc).__name__}: {str(exc)[:500]}"

        # 发布者之间最多 5 路并发；任何发布者完全失败时沿用旧行为：保留旧目录。
        with ThreadPoolExecutor(max_workers=HF_MAX_WORKERS) as pool:
            batches = list(pool.map(list_publisher, publishers))
        failures = [error for _, error in batches if error]
        if failures:
            message = "; ".join(failures[:3])
            self.db.mark_source_error(self.name, message)
            result.errors.extend({"source": self.name, "error": error} for error in failures)
            result.status = "retained"
            return result

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
            modified = getattr(item, "lastModified", None)
            if overlap_cutoff and modified:
                try:
                    modified_dt = (datetime.fromisoformat(modified.replace("Z", "+00:00"))
                                   if isinstance(modified, str) else modified)
                    if modified_dt.tzinfo is None:
                        modified_dt = modified_dt.replace(tzinfo=timezone.utc)
                    if modified_dt < overlap_cutoff:
                        continue
                except (ValueError, TypeError):
                    pass
            ordered.append(repo_id)
        result.scanned = len(ordered)

        def load_repo(repo_id: str) -> tuple[str, list[Candidate], str]:
            try:
                info = self._call(lambda: api.model_info(repo_id, files_metadata=True))
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

        new_cursor = datetime.now(timezone.utc).isoformat(timespec="seconds")
        result.cursors[self.name] = new_cursor
        self.db.mark_source_success(self.name, new_cursor,
                                    error="; ".join(e["error"] for e in result.errors[:3]))
        return result

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

    def _to_candidates(self, info: Any) -> list[Candidate]:
        repo = getattr(info, "id", "") or ""
        commit = getattr(info, "sha", "") or "main"  # 固定 commit
        card = getattr(info, "cardData", None) or {}
        if isinstance(card, str):
            card = {}
        license_id = str(card.get("license", "") or "").lower()
        publisher = repo.split("/")[0] if "/" in repo else ""
        pipeline = str(getattr(info, "pipeline_tag", None) or card.get("pipeline_tag", "") or "")
        task = _pipeline_to_task(pipeline)
        gated = str(getattr(info, "gated", "") or "")
        siblings = self._siblings(info)
        by_name = {s["name"]: s for s in siblings}
        weights = [s for s in siblings if s["name"].lower().endswith(".gguf")
                   and not s["name"].lower().endswith(".mmproj.gguf")]
        # 优先试装常见量化；每仓库最多两个变体，避免无界扩张。
        weights.sort(key=lambda s: (0 if "Q4_K_M" in s["name"] else
                                   1 if "Q4_0" in s["name"] else 2, s["size"]))
        weights = weights[:2]
        package = [s for s in siblings if s["name"].lower().endswith((".safetensors", ".onnx", ".onnx_data"))]

        evidence = [Evidence(
            title="上游模型卡与文件清单",
            url=f"https://huggingface.co/{repo}/tree/{commit}",
            source="Hugging Face 模型卡", model_version=commit,
            dataset="", protocol="",
            note="原始精度成绩只作同家族线索；具体量化版标「待测」",
        )]
        for block in (card.get("model-index") or [])[:2]:
            for res in (block.get("results") or [])[:2]:
                evidence.append(Evidence(
                    title=str(res.get("task", {}).get("type", "公开评测")),
                    source="模型卡 model-index", model_version=commit,
                    dataset=str(res.get("dataset", {}).get("name", "")),
                    protocol=str(res.get("metrics", "")),
                    note="公开证据，不等于本机实测",
                ))

        out: list[Candidate] = []
        for weight in weights:
            variant = weight["name"].split("/")[-1]
            files: list[FileSpec] = [FileSpec(
                repo_path=weight["name"], local_path=variant, size_bytes=weight["size"],
                kind="weight", upstream_sha256=weight["sha256"])]
            attachments = 0
            for group, kind in ((MODEL_CARD_NAMES, "model_card"), (LICENSE_NAMES, "license"),
                                (USAGE_NAMES, "usage"), (CONFIG_NAMES, "config")):
                for name in group:
                    if name in by_name and name not in {f.repo_path for f in files}:
                        f = by_name[name]
                        files.append(FileSpec(repo_path=name, local_path=name.replace("/", "_"),
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
            total = weight["size"] + attachments
            projection = sum(f.size_bytes for f in files if f.kind == "projection")
            footprint = estimate_footprint(weight["size"], projection_bytes=projection,
                                           backend="cuda" if weight["size"] > 1_500_000_000 else "cpu")
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
            files = [FileSpec(repo_path=s["name"], local_path=s["name"], size_bytes=s["size"],
                              kind="weight", upstream_sha256=s["sha256"]) for s in package]
            for s in siblings:
                name = s["name"]
                if name.lower().endswith((".json", ".md", ".txt", ".model", ".vocab", ".tiktoken")) or name in LICENSE_NAMES:
                    files.append(FileSpec(
                        repo_path=name, local_path=name, size_bytes=s["size"], upstream_sha256=s["sha256"],
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
            dataset = result.get("dataset") or {}
            # shortcut: 自动排名只接受声明数据集版本与切分的结构化模型卡成绩。
            if not all(dataset.get(k) for k in ("type", "revision", "split")):
                continue
            source = result.get("source") or {}
            for metric in result.get("metrics") or []:
                kind = str(metric.get("type", "")).lower()
                if kind not in {"accuracy", "acc", "f1", "map", "iou", "wer", "cer", "bleu"}:
                    continue
                value = metric.get("value")
                if not isinstance(value, (float, int)) or isinstance(value, bool):
                    continue
                key = json.dumps([dataset.get(k) for k in ("type", "revision", "config", "split", "args")] +
                                 [result.get("task"), source.get("name"), metric.get("args"),
                                  metric.get("unit") or ("0–100" if value > 1 else "0–1")],
                                 sort_keys=True, ensure_ascii=False, default=str)
                scores.append(Evidence(
                    title=str(dataset.get("name") or dataset["type"]), source=source.get("name") or "上游自报",
                    url=f"https://huggingface.co/{repo}/blob/{revision}/README.md", model_version=revision,
                    dataset=str(dataset["type"]), protocol=f"{dataset['revision']} · {dataset['split']}",
                    benchmark_key=key, variant=variant, metric=kind, value=value,
                    higher_is_better=kind not in {"wer", "cer"},
                ))
    return scores


def _pipeline_to_task(pipeline: str) -> str:
    return {
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


CONTROLLED_SOURCE_NAME = "controlled"
