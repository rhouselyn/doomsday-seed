"""本地推理子进程与资源锁（文档 §2、§3、§6）。

固定构建的 llama-server 加载本地 GGUF 路径；有验证过的 CUDA 后端时优先用独显，
CPU 为基础回退。多模态按支持情况加载匹配 mmproj。推理与评测共用资源锁，
单次只加载一个模型；正在交互时后台评测排队。服务与子进程只监听回环地址，
推理进程启动时设置 HF_HUB_OFFLINE=1 且只用本地路径；不回退到云。
"""
from __future__ import annotations

import socket
import subprocess
import threading
import time
from collections import deque
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import httpx

from .adapters import get_adapter
from .config import LOOPBACK_HOST, Paths, safe_path
from .db import Database
from .locks import LockBusy, file_lock
from .models import Candidate, HardwareSample, RuntimeConfig, Seed
from .runtime_registry import RuntimeRegistry, _offline_env


class RuntimeUnavailable(Exception):
    def __init__(self, message: str, code: str = "runtime"):
        super().__init__(message)
        self.code = code  # unregistered | waiting_resource | runtime | no_model


class ResourceManager:
    """单次只加载一个模型；交互式请求优先，后台评测排队。"""

    def __init__(self):
        self._cond = threading.Condition()
        self.loaded: str | None = None
        self.users = 0
        self.interactive = 0

    @contextmanager
    def acquire(self, model_key: str, interactive: bool, timeout: float = 180.0):
        deadline = time.time() + timeout
        with self._cond:
            while True:
                same = self.loaded in (None, model_key)
                allowed = same and (interactive or self.interactive == 0)
                if allowed:
                    break
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise RuntimeUnavailable(
                        "等待资源超时：单次只加载一个模型，正在交互时后台评测排队",
                        "waiting_resource")
                self._cond.wait(min(0.5, remaining))
            if self.loaded is None:
                self.loaded = model_key
            self.users += 1
            if interactive:
                self.interactive += 1
        try:
            yield
        finally:
            with self._cond:
                self.users -= 1
                if interactive:
                    self.interactive -= 1
                if self.users == 0 and self.interactive == 0:
                    self.loaded = None
                self._cond.notify_all()

    def state(self) -> dict[str, Any]:
        with self._cond:
            return {"loaded_model": self.loaded, "users": self.users,
                    "interactive": self.interactive}


@dataclass
class RunningServer:
    process: subprocess.Popen
    base_url: str
    backend: str
    seed_id: str
    config: RuntimeConfig
    cold_start_ms: float
    log_path: Path
    lines: deque = field(default_factory=lambda: deque(maxlen=300))

    def diagnostics(self) -> dict[str, Any]:
        text = "\n".join(self.lines)
        return {
            "backend": self.backend, "base_url": self.base_url,
            "cold_start_ms": self.cold_start_ms, "log_path": str(self.log_path),
            "log_tail": text[-4000:],
            "cuda_init_failed": any(k in text.lower() for k in
                                    ("cuda error", "out of memory", "failed to initialize cuda",
                                     "cublas", "no cuda device")),
            "context_overflow": "context size" in text.lower() and "exceed" in text.lower(),
        }


class InferenceService:
    def __init__(self, db: Database, paths: Paths, registry: RuntimeRegistry,
                 sample_provider: Callable[[], HardwareSample] | None = None):
        self.db = db
        self.paths = paths
        self.registry = registry
        self._sample_provider = sample_provider
        self._resources = ResourceManager()
        self._lock = threading.RLock()
        self._server: RunningServer | None = None
        self._runtime_lease = ExitStack()
        self._runtime_users = 0
        self._idle_timer: threading.Timer | None = None
        # 复测在暂存区进行；注册后 _load 走同一套公共接口，不绕过适配器。
        self._staging: dict[str, tuple[Path, dict[str, Any]]] = {}
        self.last_cold_start_ms: float = 0.0
        self.last_backend_note: str = ""
        (paths.data / "logs").mkdir(parents=True, exist_ok=True)

    @property
    def sample(self) -> HardwareSample:
        return self._sample_provider() if self._sample_provider else HardwareSample()

    # ---------- 模型解析 ----------
    def _load(self, model: str) -> tuple[Seed, Candidate | None, RuntimeConfig, Path]:
        seed_id = model
        if model in self._staging:
            directory, manifest = self._staging[model]
            seed = Seed(id=model, candidate_id="", rel_dir="", manifest=manifest,
                        offline_ready=True, completeness=1.0)
            return seed, None, RuntimeConfig(**(manifest.get("runtime_config") or {})), directory
        if ":" in model and not model.startswith("seed-"):
            task, profile = model.split(":", 1)
            pointer = self.db.get_default(task, profile)
            if pointer.seed_id:
                seed_id = pointer.seed_id
        seed = self.db.get_seed(seed_id)
        if seed is None:
            matches = [s for s in self.db.list_seeds()
                       if s.manifest.get("variant") == model and s.offline_ready]
            if matches:
                seed = matches[0]
        if (seed is None or not seed.offline_ready or seed.integrity != "ok"
                or seed.manifest.get("task", "chat") != "chat"):
            raise RuntimeUnavailable(
                "本机暂无可运行模型：请先完成一次下载与离线验证；断网时不回退到云。",
                "no_model")
        seed_dir = self.paths.seeds / seed.rel_dir
        manifest = seed.manifest
        cfg = RuntimeConfig(**(manifest.get("runtime_config") or {}))
        candidate = self.db.get_candidate(seed.candidate_id)
        return seed, candidate, cfg, seed_dir

    def _weight(self, seed: Seed) -> tuple[str | None, str | None]:
        weight = projection = None
        for item in seed.manifest.get("sizes_and_hashes", []):
            if item.get("kind") == "weight" and not weight:
                weight = item["relative_path"]
            if item.get("kind") == "projection" and not projection:
                projection = item["relative_path"]
        return weight, projection

    def backend_for(self, cfg: RuntimeConfig) -> str:
        if cfg.backend == "cuda" and (self.registry.is_verified("cuda")
                                      or self.sample.backend_available.get("cuda", False)):
            return "cuda"
        return "cpu"

    # ---------- 子进程 ----------
    def _free_port(self) -> int:
        with socket.socket() as sock:
            sock.bind((LOOPBACK_HOST, 0))
            return int(sock.getsockname()[1])

    def _start(self, seed: Seed, seed_dir: Path, cfg: RuntimeConfig,
               backend: str) -> RunningServer:
        command = self.registry.command(backend)
        if not command:
            raise RuntimeUnavailable(
                f"{backend} 运行器未登记：请安装锁定的 llama-server 构建，作业标「待适配」",
                "unregistered")
        port = self._free_port()
        weight, projection = self._weight(seed)
        if not weight:
            raise RuntimeUnavailable(f"清单中没有权重文件：{seed.rel_dir}", "runtime")
        try:
            weight_path = safe_path(seed_dir, weight)
            if not weight_path.is_file():
                raise ValueError(f"缺少权重文件：{weight}")
            if cfg.extra_args:
                raise ValueError("不允许清单中的额外启动参数")
        except ValueError as exc:
            raise RuntimeUnavailable(str(exc), "runtime") from exc
        args = list(command) + [
            "--host", LOOPBACK_HOST, "--port", str(port),
            "-m", str(weight_path),
            "-c", str(cfg.context), "-np", str(cfg.concurrency),
        ]
        if backend == "cuda":
            args += ["-ngl", str(cfg.gpu_layers)]
        if cfg.mmproj or projection:
            mmproj = cfg.mmproj or projection
            try:
                path = safe_path(seed_dir, mmproj)
            except ValueError as exc:
                raise RuntimeUnavailable(str(exc), "runtime") from exc
            if not path.exists():
                raise RuntimeUnavailable(f"缺少匹配的 mmproj 投影文件：{mmproj}", "runtime")
            args += ["--mmproj", str(path.resolve())]
        if cfg.chat_template:
            args += ["--chat-template", cfg.chat_template]
        args += cfg.extra_args
        log_path = self.paths.data / "logs" / f"runner-{seed.id}.log"
        started = time.perf_counter()
        try:
            proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="replace",
                                    env=_offline_env(), cwd=str(self.paths.data),
                                    bufsize=1)
        except OSError as exc:
            raise RuntimeUnavailable(f"运行器启动失败：{exc}", "runtime") from exc
        server = RunningServer(process=proc, base_url=f"http://{LOOPBACK_HOST}:{port}",
                               backend=backend, seed_id=seed.id, config=cfg,
                               cold_start_ms=0.0, log_path=log_path)

        def drain():
            with log_path.open("a", encoding="utf-8") as fh:
                for line in iter(proc.stdout.readline, ""):  # type: ignore[union-attr]
                    line = line.rstrip()
                    server.lines.append(line)
                    fh.write(line + "\n")
                    fh.flush()

        threading.Thread(target=drain, daemon=True).start()
        health = get_adapter(cfg.adapter).health_path
        deadline = time.time() + 90
        while time.time() < deadline:
            if proc.poll() is not None:
                diag = server.diagnostics()
                raise RuntimeUnavailable(
                    f"运行器冷启动退出（码 {proc.returncode}）：{diag['log_tail'][-500:]}",
                    "runtime")
            try:
                resp = httpx.get(server.base_url + health, timeout=2.0, trust_env=False)
                if resp.status_code == 200:
                    server.cold_start_ms = (time.perf_counter() - started) * 1000
                    self.last_cold_start_ms = server.cold_start_ms
                    return server
            except httpx.HTTPError:
                pass
            time.sleep(0.3)
        self._terminate(server)
        raise RuntimeUnavailable(
            f"运行器冷启动超时：{server.diagnostics()['log_tail'][-500:]}", "runtime")

    @staticmethod
    def _terminate(server: RunningServer | None) -> None:
        if server is None:
            return
        proc = server.process
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    def stop_all(self) -> None:
        with self._lock:
            if self._idle_timer:
                self._idle_timer.cancel()
                self._idle_timer = None
            self._terminate(self._server)
            self._server = None
            if not self._runtime_users:
                self._runtime_lease.close()

    @contextmanager
    def reserve(self, key: str, *, interactive=False, keep_warm=False):
        with self._resources.acquire(key, interactive):
            with self._lock:
                if self._idle_timer:
                    self._idle_timer.cancel()
                    self._idle_timer = None
                elif not self._runtime_users:
                    try:
                        self._runtime_lease.enter_context(file_lock(self.paths.data / "runtime.lock", wait=180))
                    except LockBusy as exc:
                        raise RuntimeUnavailable(str(exc), "waiting_resource") from exc
                self._runtime_users += 1
                if not keep_warm:
                    self._terminate(self._server)
                    self._server = None
            try:
                yield
            finally:
                with self._lock:
                    self._runtime_users -= 1
                    if not self._runtime_users:
                        if keep_warm and self._server:
                            timer = None

                            def expire():
                                with self._lock:
                                    if self._idle_timer is timer:
                                        self.stop_all()

                            timer = self._idle_timer = threading.Timer(15, expire)
                            timer.daemon = True
                            timer.start()
                        else:
                            self.stop_all()

    @contextmanager
    def _session(self, model: str, interactive: bool) -> Iterator[tuple[RunningServer, Seed, RuntimeConfig]]:
        # Resolve again after acquiring the cross-process lease, in case a migration
        # or revalidation changed the archive while this request was waiting.
        seed, _, _, _ = self._load(model)
        with self.reserve(seed.id, interactive=interactive, keep_warm=True):
            configured = self.db.get_settings().download_root
            if configured != self.paths.download_root:
                self.paths.set_download_root(configured)
            seed, _, cfg, seed_dir = self._load(model)
            with self._lock:
                started_new = False
                if (self._server is None or self._server.seed_id != seed.id
                        or self._server.config.fingerprint() != cfg.fingerprint()
                        or self._server.process.poll() is not None):
                    self._terminate(self._server)
                    self._server = None
                    backend = self.backend_for(cfg)
                    try:
                        self._server = self._start(seed, seed_dir, cfg, backend)
                        started_new = True
                        self.last_backend_note = f"{backend} 冷启动成功"
                        self.registry.mark_verified(backend, "本机冷启动与 /health 通过")
                    except RuntimeUnavailable as exc:
                        if backend == "cuda":
                            # §3 CUDA 初始化失败则走 CPU，并记录具体原因。
                            self.last_backend_note = f"CUDA 初始化失败，回退 CPU：{exc}"
                            self._server = self._start(seed, seed_dir, cfg, "cpu")
                            started_new = True
                        else:
                            raise
                server = self._server
                # 同一进程内的后续请求不算冷启动，冷启动单独计时。
                self.last_cold_start_ms = server.cold_start_ms if started_new else 0.0
            yield server, seed, cfg

    # ---------- 公共聊天接口 ----------
    def chat(self, model: str, messages: list[dict[str, Any]], *, stream: bool = False,
             temperature: float | None = None, max_tokens: int | None = None,
             interactive: bool = True) -> dict[str, Any]:
        with self._session(model, interactive) as (server, seed, cfg):
            adapter = get_adapter(cfg.adapter)
            payload = adapter.chat_payload(messages, stream=False, temperature=temperature,
                                           max_tokens=max_tokens, model=seed.id)
            try:
                resp = httpx.post(server.base_url + adapter.chat_path, json=payload,
                                  timeout=180.0, trust_env=False)
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPError as exc:
                raise RuntimeUnavailable(f"本地推理请求失败：{exc}", "runtime") from exc
            out = adapter.parse_response(data, model, server.backend)
            out["doomsday"]["seed_id"] = seed.id
            out["doomsday"]["runner_version"] = cfg.runner_version
            return out

    def chat_stream(self, model: str, messages: list[dict[str, Any]], *,
                    temperature: float | None = None, max_tokens: int | None = None,
                    interactive: bool = True) -> Iterator[str]:
        with self._session(model, interactive) as (server, seed, cfg):
            adapter = get_adapter(cfg.adapter)
            payload = adapter.chat_payload(messages, stream=True, temperature=temperature,
                                           max_tokens=max_tokens, model=seed.id)
            finished = False
            try:
                with httpx.stream("POST", server.base_url + adapter.chat_path,
                                  json=payload, timeout=180.0, trust_env=False) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        chunk = adapter.translate_stream(
                            line, model,
                            {"backend": server.backend,
                             "runner_version": cfg.runner_version, "seed_id": seed.id})
                        if chunk:
                            if chunk.strip() == "data: [DONE]":
                                finished = True
                            yield chunk
            except httpx.HTTPError as exc:
                raise RuntimeUnavailable(f"本地推理流失败：{exc}", "runtime") from exc
            if not finished:
                yield "data: [DONE]\n\n"

    # ---------- 暂存区复用：复测与 Agent 调试都走同一套公共接口 ----------
    def register_staging(self, seed_id: str, directory: Path, manifest: dict[str, Any]) -> None:
        self._staging[seed_id] = (Path(directory), manifest)

    def unregister_staging(self, seed_id: str) -> None:
        self._staging.pop(seed_id, None)
        with self._lock:
            if self._server and self._server.seed_id == seed_id:
                self.stop_all()

    def start_staging_server(self, seed_id: str, cfg: RuntimeConfig,
                             backend: str | None = None) -> RunningServer:
        directory, manifest = self._staging[seed_id]
        seed = Seed(id=seed_id, candidate_id="", rel_dir="", manifest=manifest,
                    offline_ready=True)
        return self._start(seed, directory, cfg, backend or self.backend_for(cfg))

    def chat_on_server(self, server: RunningServer, messages: list[dict[str, Any]],
                       max_tokens: int | None = None, stream: bool = False) -> dict[str, Any]:
        adapter = get_adapter(server.config.adapter)
        payload = adapter.chat_payload(messages, stream=stream, max_tokens=max_tokens,
                                       model=server.seed_id)
        resp = httpx.post(server.base_url + adapter.chat_path, json=payload,
                          timeout=90.0, trust_env=False)
        resp.raise_for_status()
        return adapter.parse_response(resp.json(), server.seed_id, server.backend)

    @staticmethod
    def stop_server(server: RunningServer | None) -> None:
        InferenceService._terminate(server)

    def models(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        ready = {s.id: s for s in self.db.list_seeds()
                 if s.offline_ready and s.integrity == "ok"
                 and s.manifest.get("task", "chat") == "chat"}
        for pointer in self.db.list_defaults():
            if pointer.task != "chat" or pointer.seed_id not in ready:
                continue
            out.append({
                "id": f"{pointer.task}:{pointer.profile}", "object": "model",
                "created": 0, "owned_by": "doomsday-seed",
                "doomsday": {"kind": "alias", "seed_id": pointer.seed_id,
                             "pinned": pointer.pinned, "source": pointer.source},
            })
        for seed in ready.values():
            cfg = seed.manifest.get("runtime_config", {})
            out.append({
                "id": seed.id, "object": "model", "created": 0, "owned_by": "local",
                "doomsday": {"kind": "version", "variant": seed.manifest.get("variant", ""),
                             "state": "offline_ready" if seed.offline_ready else "downloaded",
                             "backend": cfg.get("backend", "cpu"),
                             "runner_version": cfg.get("runner_version", ""),
                             "completeness": seed.completeness},
            })
        return out

    def status(self) -> dict[str, Any]:
        return {
            "resource_lock": self._resources.state(),
            "loaded": self._server.seed_id if self._server else None,
            "base_url": self._server.base_url if self._server else None,
            "last_cold_start_ms": self.last_cold_start_ms,
            "backend_note": self.last_backend_note,
            "registry_version": self.registry.version(),
        }
