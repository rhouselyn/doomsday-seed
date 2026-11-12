"""Archive a user-specified local model package without changing its source."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path

from .archive import build_manifest, verify_seed, write_manifest, write_usage
from .models import Candidate, DownloadPlan, FileSpec, HardwareSample, Job, RuntimeConfig, Seed


def enqueue(service, raw_path: str, task: str) -> Job:
    source = Path(raw_path).expanduser()
    if not source.is_absolute() or not source.exists():
        raise ValueError("请提供存在的本机绝对路径")
    if source.is_symlink() or (hasattr(source, "is_junction") and source.is_junction()):
        raise ValueError("请直接提供模型文件或目录，而不是链接")
    source = source.resolve()
    root = service.paths.data.resolve()
    if source == root or source in root.parents or root in source.parents:
        raise ValueError("请选择种子库数据目录之外的模型文件")
    job_id = uuid.uuid4().hex[:16]
    job = Job(id=job_id, kind="local_import", idempotency_key="local-import|" + job_id,
              payload={"source": str(source), "task": task}, message="等待归档本地文件")
    service.db.create_job(job)
    service.pipeline._start_worker([job.id], wait=False)
    return job


def run(pipeline, job: Job):
    from .jobs import JobResult

    db, paths = pipeline.db, pipeline.paths
    source = Path(job.payload["source"])
    base = source if source.is_dir() else source.parent
    stage = pipeline.downloader.files_dir(job.id)
    try:
        if not source.exists():
            raise ValueError("本地来源不存在")
        files = [source] if source.is_file() else []
        if source.is_dir():
            for directory, names, filenames in os.walk(source, followlinks=False):
                folder = Path(directory)
                for name in names:
                    item = folder / name
                    if item.is_symlink() or (hasattr(item, "is_junction") and item.is_junction()):
                        raise ValueError("模型目录含有链接，请选择实际文件")
                for name in filenames:
                    item = folder / name
                    if item.is_symlink() or not item.is_file():
                        raise ValueError("模型目录含有链接或无法读取的文件")
                    files.append(item)
                    if len(files) > 5000:
                        raise ValueError("模型包文件数量超过 5000")
        if not files:
            raise ValueError("所选目录没有文件")
        stage.mkdir(parents=True, exist_ok=True)
        records, digest = [], hashlib.sha256()
        for index, source_file in enumerate(sorted(files)):
            relative = "payload/" + source_file.relative_to(base).as_posix()
            destination = stage.joinpath(*relative.split("/"))
            destination.parent.mkdir(parents=True, exist_ok=True)
            before = source_file.stat()
            file_hash = hashlib.sha256()
            with source_file.open("rb") as src, destination.open("wb") as dst:
                for chunk in iter(lambda: src.read(1024 * 1024), b""):
                    file_hash.update(chunk)
                    dst.write(chunk)
            after = source_file.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError(f"来源文件在复制时发生变化：{relative}")
            sha = file_hash.hexdigest()
            digest.update((relative + "\0" + sha).encode("utf-8"))
            kind = "weight" if source_file.suffix.lower() in (
                ".gguf", ".safetensors", ".onnx", ".task", ".pt", ".pth", ".bin",
                ".ckpt", ".h5", ".tflite", ".pkl", ".zip", ".gz",
            ) else "config"
            records.append(FileSpec(repo_path=relative, local_path=relative, size_bytes=after.st_size,
                                    kind=kind, upstream_sha256=sha))
            db.update_job(job.id, state="downloading", progress=0.8 * (index + 1) / len(files),
                          message=f"正在归档本地文件 {index + 1}/{len(files)}")
        if not any(f.kind == "weight" for f in records):
            raise ValueError("未发现支持归档的模型权重")
        revision = digest.hexdigest()
        name = source.name
        candidate = Candidate(
            id=Candidate.make_id("local", "local/" + name, revision, name), source="local",
            repo="local/" + name, commit_rev=revision, variant=name, task=job.payload["task"],
            files=records, size_bytes=sum(f.size_bytes for f in records),
            rationale="用户指定的本地文件，尚未验证许可和本机适配。",
        )
        config = RuntimeConfig(adapter="archive-only")
        plan = DownloadPlan(candidate_id=candidate.id, repo=candidate.repo, commit_rev=revision,
                            files=records, runtime_config=config)
        check = pipeline.downloader.verify(stage, plan)
        manifest = build_manifest(
            seed_id=candidate.id, candidate=candidate, plan=plan, verification=check,
            runtime_config=config, evaluation=None, hardware=HardwareSample(),
            offline_check={"passed": False, "method": "本地文件校验"}, external_references=[],
        )
        manifest["collection_mode"] = "local_import"
        manifest["runner"] = {"name": "未准备运行器", "version": "", "adapter": "archive-only",
                              "backend": "", "command": []}
        write_usage(stage, "# 本地导入模型\n\n文件已复制并校验，尚未验证许可与本机试用。\n")
        write_manifest(stage, manifest)
        final = paths.seeds / candidate.id
        if final.exists():
            existing = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
            if existing.get("seed_id") != candidate.id or not existing.get("sizes_and_hashes") or not verify_seed(final, existing)[0]:
                raise ValueError("同名归档不完整，请先检查现有文件")
            manifest = existing
        else:
            os.replace(stage, final)
        with db.transaction():
            db.upsert_candidate(candidate)
            previous = db.get_seed(candidate.id)
            if previous is None:
                db.put_seed(Seed(id=candidate.id, candidate_id=candidate.id, rel_dir=candidate.id,
                                 manifest=manifest, integrity="ok", completeness=1.0, offline_ready=False))
                db.set_candidate_state(candidate.id, "downloaded", "本地文件已归档，待验证")
            db.update_job(job.id, state="done", stage="完成", progress=1.0,
                          message=f"已归档到 {final}", payload={**job.payload, "seed_id": candidate.id})
        if stage.exists() and stage.resolve().is_relative_to(paths.staging.resolve()):
            shutil.rmtree(stage)
        return JobResult(job.id, "done", "本地文件已归档", candidate.id)
    except Exception as exc:
        db.update_job(job.id, state="failed", stage="失败", message=str(exc)[:500])
        return JobResult(job.id, "failed", str(exc))
