"""固定版本下载、预检与校验（文档 §5 第 3、4 步、§6）。

用 huggingface_hub 拉取固定 commit 的精确文件清单；暂存与最终目录位于同一卷；
落盘前检查能否写完当前包；校验文件大小、上游可核验的摘要和本地 SHA-256；
校验通过后重命名，再由调用方用数据库事务更新可见状态。断线后按同一版本续跑。
"""
from __future__ import annotations

import hashlib
import os
import shutil
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .config import HF_BACKOFF_BASE_SECONDS, HF_MAX_RETRIES, Paths, safe_path
from .models import DownloadPlan, FileSpec, RuntimeConfig


class DownloadError(Exception):
    def __init__(self, message: str, code: str = "other"):
        super().__init__(message)
        self.code = code  # network | resource | verify | other


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class PreflightReport:
    ok: bool = True
    total_bytes: int = 0
    free_bytes: int = 0
    same_volume: bool = True
    files: list[dict[str, Any]] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok, "total_bytes": self.total_bytes, "free_bytes": self.free_bytes,
            "same_volume": self.same_volume, "files": self.files, "reasons": self.reasons,
        }


FetchFn = Callable[[str, str, str, Path], Path]


def hub_fetch(repo: str, filename: str, revision: str, dest: Path) -> Path:
    """默认取文件实现；hf_hub_download 自带断点续传与 .incomplete 临时文件。"""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise DownloadError("huggingface_hub 未安装，无法下载", "network") from exc
    return Path(hf_hub_download(
        repo_id=repo, filename=filename, revision=revision,
        local_dir=str(dest),
    ))


def resolve_fixed_commit(repo: str, ref: str, api: Any = None,
                         sleep: Callable[[float], None] = time.sleep) -> str:
    """§5：把分支引用解析成不可变 commit，受控候选也必须使用固定版本。"""
    import re
    if re.fullmatch(r"[0-9a-f]{7,40}", ref or ""):
        return ref
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    last: Exception | None = None
    for attempt in range(HF_MAX_RETRIES):
        try:
            info = api.model_info(repo, revision=ref, files_metadata=True)
            sha = getattr(info, "sha", "") or ""
            if not sha:
                raise DownloadError(f"未取得 {repo}@{ref} 的固定 commit", "network")
            return sha
        except Exception as exc:
            last = exc
            sleep(HF_BACKOFF_BASE_SECONDS * (2 ** attempt))
    raise DownloadError(f"解析固定 commit 失败：{last}", "network")


class Downloader:
    def __init__(self, paths: Paths, fetch: FetchFn | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.paths = paths
        self.fetch: FetchFn = fetch or hub_fetch
        self.sleep = sleep
        paths.staging.mkdir(parents=True, exist_ok=True)
        paths.seeds.mkdir(parents=True, exist_ok=True)

    # ---------- 计划 ----------
    @staticmethod
    def build_plan(candidate_id: str, repo: str, commit: str, files: list[FileSpec],
                   runtime_config: RuntimeConfig, total_bytes: int = 0) -> DownloadPlan:
        return DownloadPlan(candidate_id=candidate_id, repo=repo, commit_rev=commit,
                            files=list(files), runtime_config=runtime_config,
                            total_bytes=total_bytes or sum(f.size_bytes for f in files))

    def staging_dir(self, job_id: str) -> Path:
        return self.paths.staging / job_id

    def files_dir(self, job_id: str) -> Path:
        return self.staging_dir(job_id) / "files"

    # ---------- 预检（dry run） ----------
    def preflight(self, plan: DownloadPlan, job_id: str, *, reuse: bool = False) -> PreflightReport:
        report = PreflightReport()
        seen = set()
        for spec in plan.files:
            for name in (spec.repo_path, spec.local_path):
                try:
                    safe_path(self.files_dir(job_id), name)
                except ValueError:
                    report.ok = False
                    report.reasons.append("文件清单含不安全的路径")
            if spec.local_path.casefold() in seen or spec.local_path.casefold() in {"manifest.json", "usage.md"}:
                report.ok = False
                report.reasons.append("文件清单含重复路径或覆盖归档元数据")
            seen.add(spec.local_path.casefold())
            item = {
                "repo_path": spec.repo_path, "local_path": spec.local_path,
                "size_bytes": spec.size_bytes, "kind": spec.kind,
                "upstream_sha256": spec.upstream_sha256,
                "sha256_verifiable": bool(spec.upstream_sha256),
            }
            report.files.append(item)
            report.total_bytes += spec.size_bytes or 0
        try:
            usage = shutil.disk_usage(self.paths.staging)
            report.free_bytes = usage.free
            # 落盘前检查能否写完当前包（预留 5% 给临时文件与元数据）。
            if not reuse and report.total_bytes and usage.free < report.total_bytes * 1.05:
                report.ok = False
                report.reasons.append(
                    f"可用空间 {usage.free} 字节不足以下载并写完当前包 {report.total_bytes} 字节")
        except OSError as exc:
            report.ok = False
            report.reasons.append(f"无法读取磁盘余量：{exc}")
        try:
            report.same_volume = (os.stat(self.paths.staging).st_dev
                                  == os.stat(self.paths.seeds).st_dev)
            if not report.same_volume:
                report.ok = False
                report.reasons.append("暂存区与最终目录不在同一卷，无法原子重命名")
        except OSError:
            report.same_volume = False
        if not plan.files:
            report.ok = False
            report.reasons.append("精确文件清单为空")
        # 图文模型必须带齐匹配的投影文件。
        if any(f.kind == "weight" for f in plan.files) and plan.runtime_config.mmproj:
            if not any(f.local_path == plan.runtime_config.mmproj for f in plan.files):
                report.ok = False
                report.reasons.append("缺少匹配的 mmproj 投影文件")
        plan.preflight = report.to_dict()
        plan.total_bytes = report.total_bytes
        return report

    # ---------- 下载 ----------
    def download(self, plan: DownloadPlan, job_id: str,
                 progress: Callable[[float, str], None] | None = None) -> Path:
        target = self.files_dir(job_id)
        target.mkdir(parents=True, exist_ok=True)
        raw = self.staging_dir(job_id) / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        total = len(plan.files)
        for index, spec in enumerate(plan.files):
            dest = safe_path(target, spec.local_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            expected = spec.upstream_sha256 or spec.sha256
            if (dest.is_file() and spec.size_bytes and dest.stat().st_size == spec.size_bytes
                    and (not expected or sha256_file(dest).lower() == expected.lower())):
                if progress:
                    progress((index + 1) / max(1, total), f"续跑：已存在 {spec.local_path}")
                continue
            if progress:
                progress(index / max(1, total), f"下载 {spec.local_path}")
            last: Exception | None = None
            for attempt in range(HF_MAX_RETRIES):
                try:
                    fetch = self.fetch
                    if plan.repo.startswith("google-ai-edge/") and plan.commit_rev == "float16/1":
                        from .media import fetch
                    src = fetch(plan.repo, spec.repo_path, plan.commit_rev, raw)
                    if not Path(src).exists():
                        raise DownloadError(f"下载结果缺失：{src}", "network")
                    if Path(src).resolve().is_relative_to(raw.resolve()):
                        os.replace(src, dest)
                    else:
                        shutil.copyfile(src, dest)
                    last = None
                    break
                except DownloadError:
                    raise
                except Exception as exc:
                    last = exc
                    self.sleep(HF_BACKOFF_BASE_SECONDS * (2 ** attempt))
            if last is not None:
                raise DownloadError(f"{spec.repo_path} 下载失败：{last}", "network")
            if progress:
                progress((index + 1) / max(1, total), f"已写入 {spec.local_path}")
        return target

    # ---------- 校验 ----------
    def verify(self, files_dir: Path, plan: DownloadPlan) -> dict[str, Any]:
        result: dict[str, Any] = {"ok": True, "files": [], "errors": [], "total_bytes": 0}
        for spec in plan.files:
            path = safe_path(files_dir, spec.local_path)
            entry: dict[str, Any] = {"repo_path": spec.repo_path, "local_path": spec.local_path,
                                     "kind": spec.kind, "present": path.exists(),
                                     "size_bytes": 0, "sha256": None,
                                     "upstream_sha256": spec.upstream_sha256}
            if not path.exists():
                result["ok"] = False
                result["errors"].append(f"缺少必需文件：{spec.local_path}")
                result["files"].append(entry)
                continue
            size = path.stat().st_size
            digest = sha256_file(path)
            entry.update(size_bytes=size, sha256=digest)
            result["total_bytes"] += size
            if spec.size_bytes and size != spec.size_bytes:
                result["ok"] = False
                result["errors"].append(
                    f"{spec.local_path} 大小不符：声明 {spec.size_bytes}，实际 {size}")
            expected = spec.upstream_sha256 or spec.sha256
            if expected and digest.lower() != expected.lower():
                result["ok"] = False
                result["errors"].append(f"{spec.local_path} 上游 SHA-256 校验失败")
            result["files"].append(entry)
        if not result["ok"]:
            raise DownloadError("；".join(result["errors"]), "verify")
        return result

    # ---------- 归档重命名 ----------
    @contextmanager
    def promote(self, job_id: str, seed_id: str):
        src = self.files_dir(job_id)
        dst = safe_path(self.paths.seeds, seed_id)
        if not src.is_dir():
            raise DownloadError("暂存目录不存在，无法归档", "other")
        previous = dst.with_name(f".previous-{dst.name}")
        if previous.exists():
            raise DownloadError("发现未完成的归档替换，请先运行 verify", "other")
        if dst.exists():
            os.replace(dst, previous)
        try:
            os.replace(src, dst)
            yield dst
        except BaseException:
            if dst.exists() and not src.exists():
                os.replace(dst, src)
            if previous.exists() and not dst.exists():
                os.replace(previous, dst)
            raise
        else:
            if previous.exists():
                shutil.rmtree(previous, ignore_errors=True)

    def weight_file(plan: DownloadPlan) -> str | None:
        for spec in plan.files:
            if spec.kind == "weight":
                return spec.local_path
        return None

    def cleanup(self, job_id: str) -> None:
        shutil.rmtree(self.staging_dir(job_id), ignore_errors=True)
