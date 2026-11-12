"""Windows 定时同步（文档 §2、§6）。

首次设置时注册任务计划程序，每天本地时间 03:00（可改）；配置错过后可补跑、
同一计划忽略重复实例。记录上次尝试、上次成功、下一次运行和跳过原因。
应用层仍用 SyncLock 处理手动触发和计划冲突；非 Windows 明确说明不支持，不假装注册成功。
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, time as dt_time, timedelta
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from .config import Paths
from .db import Database

TASK_NAME = "DoomsdaySeedDailySync"


def _local_now() -> datetime:
    return datetime.now().astimezone().replace(tzinfo=None)


def parse_schedule_time(value: str) -> dt_time:
    try:
        hour, minute = (int(part) for part in str(value).split(":")[:2])
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return dt_time(hour, minute)
    except (ValueError, TypeError):
        pass
    return dt_time(3, 0)


def next_run(schedule_time: str, now: datetime | None = None) -> str:
    now = now or _local_now()
    clock = parse_schedule_time(schedule_time)
    candidate = now.replace(hour=clock.hour, minute=clock.minute, second=0, microsecond=0)
    if candidate <= now:
        candidate += timedelta(days=1)
    return candidate.isoformat(timespec="minutes")


def task_xml(schedule_time: str, python_exe: str, project_root: Path,
             now: datetime | None = None) -> str:
    now = now or _local_now()
    clock = parse_schedule_time(schedule_time)
    start = now.replace(hour=clock.hour, minute=clock.minute, second=0, microsecond=0)
    if start <= now:
        start += timedelta(days=1)
    return f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>末日种子：每日目录同步（Agent 不可用时仍刷新目录并使用受控候选）</Description>
    <URI>\\{TASK_NAME}</URI>
  </RegistrationInfo>
  <Triggers>
    <CalendarTrigger>
      <StartBoundary>{start.isoformat(timespec='seconds')}</StartBoundary>
      <Enabled>true</Enabled>
      <ScheduleByDay><DaysInterval>1</DaysInterval></ScheduleByDay>
    </CalendarTrigger>
  </Triggers>
  <Settings>
    <StartWhenAvailable>true</StartWhenAvailable>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <ExecutionTimeLimit>PT2H</ExecutionTimeLimit>
    <Enabled>true</Enabled>
    <Priority>7</Priority>
  </Settings>
  <Principals>
    <Principal id="Author"><LogonType>InteractiveToken</LogonType><RunLevel>LeastPrivilege</RunLevel></Principal>
  </Principals>
  <Actions Context="Author">
    <Exec>
      <Command>{escape(python_exe)}</Command>
      <Arguments>-m doomsday_seed sync --scheduled</Arguments>
      <WorkingDirectory>{escape(str(project_root))}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"""


class ScheduleService:
    def __init__(self, db: Database, paths: Paths, python_exe: str | None = None):
        self.db = db
        self.paths = paths
        self.python_exe = python_exe or sys.executable

    @property
    def supported(self) -> bool:
        return os.name == "nt"

    def _schtasks(self, args: list[str]) -> tuple[bool, str]:
        try:
            result = subprocess.run(["schtasks", *args], capture_output=True,
                                    text=True, timeout=30, encoding="utf-8", errors="replace")
            return result.returncode == 0, (result.stdout + result.stderr).strip()[:1000]
        except (OSError, subprocess.TimeoutExpired) as exc:
            return False, str(exc)

    def register(self, schedule_time: str) -> dict[str, Any]:
        if not self.supported:
            return {"registered": False, "reason": "当前系统不是 Windows，任务计划程序不可用；"
                                                   "请用系统自带的定时机制调用 CLI 同步命令"}
        xml_path = self.paths.data / "daily-sync-task.xml"
        xml_path.parent.mkdir(parents=True, exist_ok=True)
        xml_path.write_text(task_xml(schedule_time, self.python_exe, self.paths.root),
                            encoding="utf-16")
        ok, detail = self._schtasks(["/Create", "/TN", TASK_NAME, "/XML", str(xml_path), "/F"])
        self.db.set_meta("schedule_registered", "1" if ok else "0")
        self.db.set_meta("schedule_time", schedule_time)
        self.db.set_meta("schedule_detail", detail)
        return {"registered": ok, "task_name": TASK_NAME, "schedule_time": schedule_time,
                "xml_path": str(xml_path), "detail": detail,
                "note": "需先以 pip install -e . 安装应用，计划任务才能调用 CLI"}

    def unregister(self) -> dict[str, Any]:
        if not self.supported:
            return {"registered": False, "reason": "非 Windows 系统"}
        ok, detail = self._schtasks(["/Delete", "/TN", TASK_NAME, "/F"])
        self.db.set_meta("schedule_registered", "0")
        return {"registered": False, "detail": detail}

    def apply_settings(self, settings) -> dict[str, Any]:
        """PUT /api/settings 后让实际生效状态与设置一致。"""
        if not settings.schedule_enabled:
            return self.unregister()
        return self.register(settings.schedule_time)

    def status(self) -> dict[str, Any]:
        settings = self.db.get_settings()
        runs = self.db.list_sync_runs(limit=20)
        last_success = self.db.get_meta("last_sync_success")
        # 状态读取只查本地记录，不主动启动系统命令；系统实际状态由显式核对接口确认。
        registered = self.db.get_meta("schedule_registered") == "1"
        query_detail = ("如需核对任务计划程序中的实际注册状态，请运行 "
                        "`doomsday-seed schedule status --verify`" if self.supported
                        else "非 Windows 系统，未注册任务计划")
        planned = next_run(settings.schedule_time)
        catch_up_due = False
        if last_success and settings.auto_update and settings.schedule_enabled:
            try:
                last = datetime.fromisoformat(last_success.replace("Z", "+00:00"))
                if last.tzinfo is not None:
                    last = last.astimezone().replace(tzinfo=None)
                today = _local_now().replace(hour=0, minute=0, second=0, microsecond=0)
                clock = parse_schedule_time(settings.schedule_time)
                due_at = today.replace(hour=clock.hour, minute=clock.minute)
                catch_up_due = last < due_at <= _local_now()
            except ValueError:
                catch_up_due = False
        return {
            "supported": self.supported,
            "registered": registered,
            "task_name": TASK_NAME,
            "schedule_time": settings.schedule_time,
            "auto_update": settings.auto_update,
            "schedule_enabled": settings.schedule_enabled,
            "last_attempt": self.db.get_meta("last_sync_attempt"),
            "last_success": last_success or None,
            "next_run": planned,
            "skip_reason": self.db.get_meta("last_sync_skip_reason"),
            "catch_up_due": catch_up_due,
            "recent_runs": runs[:5],
            "query_detail": query_detail[:500],
        }

    def verify_registration(self) -> dict[str, Any]:
        """显式核对任务计划程序中的实际状态（只有用户请求时才调用系统命令）。"""
        if not self.supported:
            return {"registered": False, "reason": "非 Windows 系统"}
        ok, detail = self._schtasks(["/Query", "/TN", TASK_NAME])
        self.db.set_meta("schedule_registered", "1" if ok else "0")
        return {"registered": ok, "task_name": TASK_NAME, "detail": detail}

    def maybe_catch_up(self, pipeline) -> dict[str, Any]:
        """配置错过后可补跑；同一计划忽略重复实例由 SyncLock 保证。"""
        state = self.status()
        if not state["catch_up_due"]:
            return {"triggered": False, "reason": "无需补跑或自动更新未启用"}
        report = pipeline.sync(trigger="catchup")
        return {"triggered": True, "report": report.model_dump()}
