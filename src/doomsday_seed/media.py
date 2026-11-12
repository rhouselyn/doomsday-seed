"""Reviewed model and runtime recipes for offline image landmarks."""
from __future__ import annotations

import base64
import binascii
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from threading import Lock

import httpx
from PIL import Image

from .config import Paths
from .models import Candidate, Evidence, FileSpec
from .runtime_registry import _offline_env

VERSION = "0.10.32"
MAX_IMAGE_BYTES = 8 * 1024 * 1024
IMAGE_BASE64_LIMIT = ((MAX_IMAGE_BYTES + 2) // 3) * 4
ASSETS = {
    "pose": ("pose_landmarker_lite", 5777746, "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a"),
    "hand": ("hand_landmarker", 7819105, "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"),
}
_install = Lock()


def candidates() -> list[Candidate]:
    models = []
    for task, (name, size, digest) in ASSETS.items():
        repo, revision, file = f"google-ai-edge/{name}", "float16/1", name + ".task"
        model_url = f"https://developers.google.com/edge/mediapipe/solutions/vision/{task}_landmarker"
        models.append(Candidate(
            id=Candidate.make_id("mediapipe", repo, revision, file), source="mediapipe",
            repo=repo, commit_rev=revision, variant=file, task=task, publisher="Google AI Edge",
            license="apache-2.0", size_bytes=size,
            files=[FileSpec(repo_path=file, local_path=file, size_bytes=size, upstream_sha256=digest)],
            evidence=[Evidence(title="官方模型与使用文档", url=model_url, source="Google AI Edge", model_version=revision)],
            rationale="已接入的官方轻量版本，适合本地图片关键点识别。",
        ))
    return models


def fetch(repo: str, filename: str, revision: str, dest: Path) -> Path:
    item = next((c for c in candidates() if (c.repo, c.variant, c.commit_rev) == (repo, filename, revision)), None)
    if item is None:
        raise ValueError("未登记的图像权重")
    name = ASSETS[item.task][0]
    url = f"https://storage.googleapis.com/mediapipe-models/{item.task}_landmarker/{name}/{revision}/{filename}"
    path = dest / filename
    with httpx.stream("GET", url, timeout=30) as response:
        response.raise_for_status()
        with path.open("wb") as file:
            total = 0
            for chunk in response.iter_bytes():
                total += len(chunk)
                if total > item.size_bytes:
                    raise ValueError("权重文件超过登记大小")
                file.write(chunk)
    return path


def runtime_dir(paths: Paths) -> Path:
    return paths.runtimes / ("mediapipe-" + VERSION)


def prepare(paths: Paths) -> None:
    target = runtime_dir(paths)
    with _install:
        if (target / ".ready").is_file():
            return
        if (sys.platform != "win32" or platform.machine().lower() not in ("amd64", "x86_64")
                or sys.version_info[:2] != (3, 13)):
            raise RuntimeError("动作运行环境暂支持 Windows x64 / Python 3.13")
        lock = Path(__file__).with_name("resources") / "mediapipe-win-cp313.txt"
        with tempfile.TemporaryDirectory(dir=paths.runtimes) as stage:
            package = Path(stage) / "packages"
            cmd = [sys.executable, "-m", "pip", "--isolated", "install", "--disable-pip-version-check",
                   "--require-hashes", "--only-binary=:all:", "--retries", "1", "--timeout", "20",
                   "--target", str(package), "-r", str(lock)]
            wheelhouse = paths.data / "wheelhouse"
            result = (subprocess.run(cmd + ["--no-index", "--find-links", str(wheelhouse)],
                                     capture_output=True, timeout=180)
                      if wheelhouse.is_dir() else None)
            if result is None or result.returncode:
                result = subprocess.run(cmd + ["--index-url", "https://pypi.org/simple"],
                                        capture_output=True, timeout=1800)
            if result.returncode:
                raise RuntimeError("动作运行环境暂未准备好，稍后会再次尝试")
            (package / ".ready").touch()
            try:
                package.rename(target)
            except OSError:
                if not (target / ".ready").is_file():
                    raise


def detect(paths: Paths, task: str, weight: Path, image: bytes) -> dict:
    runtime = runtime_dir(paths)
    if not (runtime / ".ready").is_file():
        raise RuntimeError("动作运行环境仍在准备")
    with tempfile.TemporaryDirectory() as folder:
        image_path = Path(folder) / "input"
        image_path.write_bytes(image)
        proc = subprocess.run(
            [sys.executable, "-I", str(Path(__file__).with_name("media_worker.py")),
             str(runtime), task, str(weight), str(image_path)],
            env=_offline_env(), capture_output=True, timeout=45,
        )
    if proc.returncode:
        raise ValueError("无法处理图片。请使用 PNG、JPEG 或 WebP，最多 1600 万像素")
    return json.loads(proc.stdout)


def decode_image(encoded: str) -> bytes:
    try:
        image = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("图片编码无效") from exc
    if not image or len(image) > MAX_IMAGE_BYTES:
        raise ValueError("图片须在 8 MiB 以内")
    from io import BytesIO
    try:
        with Image.open(BytesIO(image)) as picture:
            if picture.format not in ("PNG", "JPEG", "WEBP") or picture.width * picture.height > 16_000_000:
                raise ValueError("不支持的图片格式或尺寸")
            picture.verify()
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError("无法读取图片") from exc
    return image


def smoke(paths: Paths, task: str, weight: Path) -> None:
    from io import BytesIO
    sample = BytesIO()
    Image.new("RGB", (128, 128)).save(sample, format="PNG")
    if detect(paths, task, weight, sample.getvalue()) != {"landmarks": []}:
        raise RuntimeError("本地关键点空图检查失败")
