"""应用组装：网页与 CLI 共用同一套业务代码（文档 §2、§6）。"""
from __future__ import annotations

import os
from datetime import datetime
from threading import Event, Thread
from typing import Any, Callable

from fastapi import Depends, FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .agent import AgentGateway, DeepSeekClient, HttpAgentClient, LocalRuleAgent
from .archive import reconcile
from .config import APP_VERSION, Paths
from .db import Database
from .debug_tools import DebugToolbox
from .discovery import HuggingFaceSource
from .downloader import Downloader
from .evaluation import Evaluator
from .hardware import probe as probe_hardware
from .jobs import JobPipeline
from .locks import LockBusy
from .models import HardwareSample
from .preselect import PreSelector, controlled_candidates
from .recovery import RecoveryService
from .runner import InferenceService, RuntimeUnavailable
from .runtime_registry import RuntimeRegistry
from .scheduler import ScheduleService, parse_schedule_time
from .security import LocalSecurity
from .credentials import CredentialStore
from . import media


class Service:
    """所有组件的唯一容器。测试可注入 api / fetch / agent_client / commit_resolver。"""

    def __init__(self, data_dir: str | os.PathLike | None = None, *,
                 hf_api: Any | None = None, fetch: Callable | None = None,
                 agent_client: Any | None = None,
                 commit_resolver: Callable[[str, str], str] | None = None):
        self.paths = Paths.detect(data_dir)
        self.paths.ensure()
        self.db = Database(self.paths.db)
        self.paths.set_download_root(self.db.get_settings().download_root)
        for candidate in [*controlled_candidates(), *media.candidates()]:
            if self.db.get_candidate(candidate.id) is None:
                self.db.upsert_candidate(candidate)
        self.registry = RuntimeRegistry(self.paths)
        self.security = LocalSecurity(self.paths, self.db)
        self.credentials = CredentialStore(self.paths.data)
        self._injected_agent = agent_client is not None
        if agent_client is None and self.credentials.get():
            agent_client = self.build_agent_client()
        elif agent_client is None and os.environ.get("DOOMSDAY_AGENT_URL"):
            agent_client = HttpAgentClient()
        self.gateway = AgentGateway(agent_client or LocalRuleAgent(),
                                    enabled=self.db.get_settings().agent_enabled)
        self.source = HuggingFaceSource(self.db, api=hf_api)
        self.inference = InferenceService(self.db, self.paths, self.registry, self.hardware)
        self.downloader = Downloader(self.paths, fetch=fetch)
        self.debug_box = DebugToolbox(self.db, self.paths, self.downloader,
                                      self.inference, self.registry)
        self.evaluator = Evaluator(self.db, self.paths, self.inference, self.registry,
                                   sample_provider=self.hardware)
        self.preselector = PreSelector(self.db, self.paths, self.gateway,
                                       commit_resolver=commit_resolver, source=self.source)
        self.pipeline = JobPipeline(
            db=self.db, paths=self.paths, registry=self.registry, source=self.source,
            preselector=self.preselector, downloader=self.downloader,
            debug_box=self.debug_box, evaluator=self.evaluator,
            inference=self.inference, gateway=self.gateway)
        self.scheduler = ScheduleService(self.db, self.paths)
        self.recovery = RecoveryService(self.db, self.paths, self.pipeline.library_lock)
        self.startup_report: dict[str, Any] = {}
        self._stop_daily = Event()
        self._daily: Thread | None = None

    def hardware(self) -> HardwareSample:
        return probe_hardware(self.paths, self.registry)

    def build_agent_client(self):
        """按当前设置创建模型服务客户端；无密钥时退回内置规则 Agent。"""
        settings = self.db.get_settings()
        if self.credentials.get():
            return DeepSeekClient(self.credentials,
                                  base_url=settings.agent_base_url,
                                  model=settings.agent_model)
        if os.environ.get("DOOMSDAY_AGENT_URL"):
            return HttpAgentClient()
        return LocalRuleAgent()

    def reload_agent_client(self):
        if not self._injected_agent:
            self.gateway.client = self.build_agent_client()

    def startup(self) -> dict[str, Any]:
        # §5 启动时对账：接回有效包，残缺包继续排队。
        try:
            with self.pipeline.library_lock(), self.inference.reserve("reconcile"):
                self.startup_report = reconcile(self.db, self.paths)
        except (LockBusy, RuntimeUnavailable):
            self.startup_report = {"skipped": "种子库正在使用"}
        self.gateway.enabled = self.db.get_settings().agent_enabled
        self.pipeline.run_pending()
        try:
            self.startup_report["catch_up"] = self.scheduler.maybe_catch_up(self.pipeline)
        except Exception as exc:
            self.startup_report["catch_up_error"] = str(exc)
        return self.startup_report

    def shutdown(self) -> None:
        self._stop_daily.set()
        if self._daily and self._daily.is_alive():
            self._daily.join()
        self.pipeline.shutdown()
        self.inference.stop_all()
        self.db.close()

    def start_daily(self) -> None:
        def loop():
            while not self._stop_daily.wait(15):
                settings = self.db.get_settings()
                if not settings.auto_update or not settings.schedule_enabled or settings.simulated_offline:
                    continue
                now = datetime.now().astimezone()
                clock = parse_schedule_time(settings.schedule_time)
                last = self.db.get_meta("last_sync_attempt")
                try:
                    attempted_today = bool(last and datetime.fromisoformat(last).astimezone().date() == now.date())
                except ValueError:
                    attempted_today = False
                if now.time() >= clock and not attempted_today:
                    self.pipeline.sync(trigger="scheduled")

        self._daily = Thread(target=loop, name="seed-daily", daemon=True)
        self._daily.start()


def create_app(service: Service | None = None):
    from contextlib import asynccontextmanager

    from .api import router
    from .security import install_middleware, require_local

    svc = service or Service()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        svc.startup()
        svc.start_daily()
        try:
            yield
        finally:
            svc.shutdown()

    def _require_local(request: Request):
        return require_local(request, svc.security)

    app = FastAPI(title="末日种子 · 本地服务", version=APP_VERSION,
                  lifespan=lifespan, dependencies=[Depends(_require_local)])
    app.state.service = svc
    install_middleware(app, svc.security)

    web = svc.paths.web
    if (web / "assets").exists():
        app.mount("/assets", StaticFiles(directory=str(web / "assets")), name="assets")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(str(web / "index.html"), media_type="text/html; charset=utf-8")

    app.include_router(router)
    return app
