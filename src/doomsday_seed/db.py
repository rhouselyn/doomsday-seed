"""SQLite 持久化：候选、订阅、作业、评测、种子、默认版本（文档 §2 / §5）。

设计约束：
* WAL + 单连接 + 可重入锁，网页与 CLI 共用同一套业务代码；
* 归档目录与程序源码分开，种子表里只存相对路径；
* 可见状态的切换（校验通过 → 重命名 → 事务更新）在调用方用 transaction() 包住。
"""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .models import (
    Candidate, DefaultPointer, EvaluationRecord, Evidence, FileSpec,
    Estimate, HardwareSample, Job, Seed, Settings, SyncReport, utc_now,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS hardware_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT, sampled_at TEXT NOT NULL, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sources (
    name TEXT PRIMARY KEY, cursor TEXT, last_success TEXT, last_error TEXT, updated_at TEXT);
CREATE TABLE IF NOT EXISTS candidates (
    id TEXT PRIMARY KEY, source TEXT NOT NULL, repo TEXT NOT NULL,
    commit_rev TEXT NOT NULL, variant TEXT NOT NULL, task TEXT NOT NULL,
    publisher TEXT NOT NULL DEFAULT '', license TEXT NOT NULL DEFAULT '',
    size_bytes INTEGER NOT NULL DEFAULT 0, files TEXT NOT NULL DEFAULT '[]',
    evidence TEXT NOT NULL DEFAULT '[]', estimate TEXT NOT NULL DEFAULT '{}',
    rationale TEXT NOT NULL DEFAULT '', state TEXT NOT NULL DEFAULT 'candidate',
    reason TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
    UNIQUE(source, repo, commit_rev, variant));
CREATE INDEX IF NOT EXISTS idx_candidates_task ON candidates(task, state);
CREATE TABLE IF NOT EXISTS subscriptions (
    task TEXT NOT NULL, profile TEXT NOT NULL, enabled INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY(task, profile));
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY, kind TEXT NOT NULL, candidate_id TEXT NOT NULL DEFAULT '',
    idempotency_key TEXT UNIQUE, state TEXT NOT NULL, stage TEXT NOT NULL,
    progress REAL NOT NULL DEFAULT 0, message TEXT NOT NULL DEFAULT '',
    payload TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state);
CREATE TABLE IF NOT EXISTS evaluations (
    id TEXT PRIMARY KEY, candidate_id TEXT NOT NULL, task TEXT NOT NULL,
    eval_version TEXT NOT NULL, backend TEXT NOT NULL, runner_version TEXT NOT NULL,
    hardware_fingerprint TEXT NOT NULL, params TEXT NOT NULL DEFAULT '{}',
    quality REAL NOT NULL DEFAULT 0, passed_cases INTEGER NOT NULL DEFAULT 0,
    total_cases INTEGER NOT NULL DEFAULT 0, median_latency_ms REAL NOT NULL DEFAULT 0,
    ttft_ms REAL NOT NULL DEFAULT 0, tokens_per_s REAL NOT NULL DEFAULT 0,
    peak_ram_mib REAL NOT NULL DEFAULT 0, peak_vram_mib REAL NOT NULL DEFAULT 0,
    cold_start_ms REAL NOT NULL DEFAULT 0, api_ok INTEGER NOT NULL DEFAULT 0,
    offline_ok INTEGER NOT NULL DEFAULT 0, score_speed REAL NOT NULL DEFAULT 0,
    score_balanced REAL NOT NULL DEFAULT 0, score_accuracy REAL NOT NULL DEFAULT 0,
    detail TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_eval_candidate ON evaluations(candidate_id, eval_version);
CREATE TABLE IF NOT EXISTS seeds (
    id TEXT PRIMARY KEY, candidate_id TEXT NOT NULL, rel_dir TEXT NOT NULL,
    manifest TEXT NOT NULL DEFAULT '{}', completeness REAL NOT NULL DEFAULT 0,
    offline_ready INTEGER NOT NULL DEFAULT 0, integrity TEXT NOT NULL DEFAULT 'unchecked',
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS defaults (
    task TEXT NOT NULL, profile TEXT NOT NULL, seed_id TEXT,
    pinned INTEGER NOT NULL DEFAULT 0, previous_seed_id TEXT,
    source TEXT NOT NULL DEFAULT 'auto', updated_at TEXT NOT NULL,
    PRIMARY KEY(task, profile));
CREATE TABLE IF NOT EXISTS sync_runs (
    id TEXT PRIMARY KEY, started_at TEXT NOT NULL, finished_at TEXT,
    status TEXT NOT NULL, trigger TEXT NOT NULL, detail TEXT NOT NULL DEFAULT '{}');
CREATE TABLE IF NOT EXISTS api_keys (
    key_hash TEXT PRIMARY KEY, label TEXT NOT NULL, created_at TEXT NOT NULL);
"""

_JSON_FIELDS: dict[str, set[str]] = {
    "candidates": {"files", "evidence", "estimate"},
    "jobs": {"payload"},
    "evaluations": {"params", "detail"},
    "seeds": {"manifest"},
    "sync_runs": {"detail"},
}


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _loads(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


class Database:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._in_transaction = False
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30)
        self.conn.row_factory = sqlite3.Row
        with self._lock:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA foreign_keys=ON")
            self.conn.execute("PRAGMA busy_timeout=10000")
        self.initialize()

    # ---------- 基础 ----------
    def initialize(self) -> None:
        with self._lock:
            self.conn.executescript(SCHEMA)
            self.conn.commit()

    def execute(self, sql: str, params: tuple | list = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self.conn.execute(sql, params)
            # 处于外层事务时不提前提交，保证「重命名 → 事务更新可见状态」的原子性。
            if not self._in_transaction:
                self.conn.commit()
            return cur

    def query(self, sql: str, params: tuple | list = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def query_one(self, sql: str, params: tuple | list = ()) -> dict | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """§5 校验通过后重命名，再用数据库事务更新可见状态。"""
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            self._in_transaction = True
            try:
                yield self.conn
            except Exception:
                self.conn.rollback()
                raise
            else:
                self.conn.commit()
            finally:
                self._in_transaction = False

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    # ---------- 设置 / 元数据 ----------
    def get_settings(self) -> Settings:
        rows = self.query("SELECT key, value FROM settings")
        known = set(Settings.model_fields)
        data = {r["key"]: _loads(r["value"], None) for r in rows if r["key"] in known}
        return Settings(**{k: v for k, v in data.items() if v is not None})

    def put_settings(self, settings: Settings) -> Settings:
        with self.transaction() as conn:
            for key, value in settings.model_dump().items():
                conn.execute(
                    "INSERT INTO settings(key, value) VALUES(?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (key, _dumps(value)),
                )
            existing = {(r[0], r[1]) for r in conn.execute(
                "SELECT task, profile FROM subscriptions WHERE enabled=1")}
            wanted = set(settings.active_pairs())
            for pair in wanted - existing:
                conn.execute("INSERT INTO subscriptions(task, profile, enabled) VALUES(?, ?, 1) "
                             "ON CONFLICT(task, profile) DO UPDATE SET enabled=1", pair)
            for pair in existing - wanted:
                conn.execute("UPDATE subscriptions SET enabled=0 WHERE task=? AND profile=?", pair)
        return self.get_settings()

    def list_subscriptions(self) -> list[dict]:
        return self.query("SELECT task, profile, enabled FROM subscriptions ORDER BY task, profile")

    def get_meta(self, key: str, default: str = "") -> str:
        row = self.query_one("SELECT value FROM meta WHERE key=?", (key,))
        return row["value"] if row else default

    def set_meta(self, key: str, value: str) -> None:
        self.execute("INSERT INTO meta(key, value) VALUES(?, ?) "
                     "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))

    # ---------- 硬件采样 ----------
    def save_sample(self, sample: HardwareSample) -> None:
        self.execute("INSERT INTO hardware_samples(sampled_at, payload) VALUES(?, ?)",
                     (sample.sampled_at, sample.model_dump_json()))
        self.execute("DELETE FROM hardware_samples WHERE id NOT IN "
                     "(SELECT id FROM hardware_samples ORDER BY id DESC LIMIT 50)")

    def latest_sample(self) -> HardwareSample | None:
        row = self.query_one("SELECT payload FROM hardware_samples ORDER BY id DESC LIMIT 1")
        return HardwareSample.model_validate_json(row["payload"]) if row else None

    # ---------- 来源游标 ----------
    def get_source(self, name: str) -> dict:
        row = self.query_one("SELECT * FROM sources WHERE name=?", (name,))
        return dict(row) if row else {"name": name, "cursor": None, "last_success": None,
                                     "last_error": None, "updated_at": None}

    def mark_source_success(self, name: str, cursor: str | None, error: str = "") -> None:
        self.execute(
            "INSERT INTO sources(name, cursor, last_success, last_error, updated_at) "
            "VALUES(?, ?, ?, ?, ?) ON CONFLICT(name) DO UPDATE SET "
            "cursor=excluded.cursor, last_success=excluded.last_success, "
            "last_error=excluded.last_error, updated_at=excluded.updated_at",
            (name, cursor, utc_now(), error, utc_now()),
        )

    def mark_source_error(self, name: str, error: str) -> None:
        """单源出错保留旧结果与旧游标。"""
        self.execute(
            "INSERT INTO sources(name, cursor, last_success, last_error, updated_at) "
            "VALUES(?, NULL, NULL, ?, ?) ON CONFLICT(name) DO UPDATE SET "
            "last_error=excluded.last_error, updated_at=excluded.updated_at",
            (name, error[:2000], utc_now()),
        )

    # ---------- 候选 ----------
    @staticmethod
    def _candidate_from_row(row: dict) -> Candidate:
        data = dict(row)
        data["files"] = [FileSpec(**f) for f in _loads(data.get("files"), [])]
        data["evidence"] = [Evidence(**e) for e in _loads(data.get("evidence"), [])]
        data["estimate"] = Estimate(**_loads(data.get("estimate"), {}))
        return Candidate(**data)

    def upsert_candidate(self, candidate: Candidate) -> Candidate:
        existing = self.get_candidate(candidate.id)
        if existing:
            # 只覆盖元数据与预估，不回退已取得的归档状态。
            candidate.state = existing.state
            candidate.reason = existing.reason
            candidate.created_at = existing.created_at
        candidate.updated_at = utc_now()
        d = candidate.model_dump()
        self.execute(
            "INSERT INTO candidates(id, source, repo, commit_rev, variant, task, publisher, "
            "license, size_bytes, files, evidence, estimate, rationale, state, reason, "
            "created_at, updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET license=excluded.license, size_bytes=excluded.size_bytes, "
            "files=excluded.files, evidence=excluded.evidence, estimate=excluded.estimate, "
            "rationale=excluded.rationale, state=excluded.state, reason=excluded.reason, "
            "updated_at=excluded.updated_at",
            (d["id"], d["source"], d["repo"], d["commit_rev"], d["variant"], d["task"],
             d["publisher"], d["license"], d["size_bytes"],
             _dumps([f.model_dump() for f in candidate.files]),
             _dumps([e.model_dump() for e in candidate.evidence]),
             _dumps(candidate.estimate.model_dump()), d["rationale"], d["state"], d["reason"],
             d["created_at"], d["updated_at"]),
        )
        return candidate

    def get_candidate(self, candidate_id: str) -> Candidate | None:
        row = self.query_one("SELECT * FROM candidates WHERE id=?", (candidate_id,))
        return self._candidate_from_row(row) if row else None

    def find_candidate_by_key(self, key: str) -> Candidate | None:
        source, repo, commit, variant = key.split("|", 3)
        row = self.query_one(
            "SELECT * FROM candidates WHERE source=? AND repo=? AND commit_rev=? AND variant=?",
            (source, repo, commit, variant))
        return self._candidate_from_row(row) if row else None

    def list_candidates(self, task: str | None = None, state: str | None = None) -> list[Candidate]:
        sql = "SELECT * FROM candidates WHERE 1=1"
        params: list[Any] = []
        if task:
            sql += " AND task=?"
            params.append(task)
        if state:
            sql += " AND state=?"
            params.append(state)
        sql += " ORDER BY updated_at DESC, repo"
        return [self._candidate_from_row(r) for r in self.query(sql, params)]

    def set_candidate_state(self, candidate_id: str, state: str, reason: str = "") -> None:
        self.execute("UPDATE candidates SET state=?, reason=?, updated_at=? WHERE id=?",
                     (state, reason, utc_now(), candidate_id))

    # ---------- 作业 ----------
    @staticmethod
    def _job_from_row(row: dict) -> Job:
        data = dict(row)
        data["payload"] = _loads(data.get("payload"), {})
        return Job(**data)

    def create_job(self, job: Job) -> tuple[Job, bool]:
        """返回 (作业, 是否新建)。幂等键命中且作业未终结时复用，保证跨档只下载一次。"""
        if job.idempotency_key:
            row = self.query_one("SELECT * FROM jobs WHERE idempotency_key=?", (job.idempotency_key,))
            if row:
                found = self._job_from_row(row)
                if found.state == "failed":
                    # 失败作业按同一版本重试，不新建重复下载。
                    self.update_job(found.id, state="queued", stage="排队", progress=0.0,
                                    message="失败作业按同一版本重新排队")
                    found = self.get_job(found.id) or found
                return found, False
        self.execute(
            "INSERT INTO jobs(id, kind, candidate_id, idempotency_key, state, stage, progress, "
            "message, payload, created_at, updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (job.id, job.kind, job.candidate_id, job.idempotency_key or None, job.state,
             job.stage, job.progress, job.message, _dumps(job.payload),
             job.created_at, job.updated_at))
        return job, True

    def update_job(self, job_id: str, **fields: Any) -> None:
        if not fields:
            return
        if "payload" in fields:
            fields["payload"] = _dumps(fields["payload"])
        fields["updated_at"] = utc_now()
        allowed = {"kind", "candidate_id", "state", "stage", "progress", "message",
                   "payload", "updated_at"}
        fields = {k: v for k, v in fields.items() if k in allowed}
        sql = "UPDATE jobs SET " + ", ".join(f"{k}=?" for k in fields) + " WHERE id=?"
        self.execute(sql, (*fields.values(), job_id))

    def get_job(self, job_id: str) -> Job | None:
        row = self.query_one("SELECT * FROM jobs WHERE id=?", (job_id,))
        return self._job_from_row(row) if row else None

    def list_jobs(self, limit: int = 100, state: str | None = None) -> list[Job]:
        sql = "SELECT * FROM jobs"
        params: list[Any] = []
        if state:
            sql += " WHERE state=?"
            params.append(state)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        return [self._job_from_row(r) for r in self.query(sql, params)]

    def recover_jobs(self) -> list[Job]:
        """§5 重启后从 SQLite 持久化作业恢复；残缺包继续排队。"""
        active = ("downloading", "verifying", "debugging", "retesting", "queued",
                  "waiting_network", "waiting_resource")
        placeholders = ",".join("?" for _ in active)
        rows = self.query(f"SELECT * FROM jobs WHERE state IN ({placeholders}) ORDER BY created_at", active)
        jobs = [self._job_from_row(r) for r in rows]
        for job in jobs:
            if job.state != "queued":
                self.update_job(job.id, state="queued", stage="排队", progress=0.0,
                                message="应用重启，作业已重新排队，未完成文件按同一版本续跑")
        return jobs

    # ---------- 评测 ----------
    def add_evaluation(self, record: EvaluationRecord) -> None:
        d = record.model_dump()
        self.execute(
            "INSERT INTO evaluations(id, candidate_id, task, eval_version, backend, runner_version, "
            "hardware_fingerprint, params, quality, passed_cases, total_cases, median_latency_ms, "
            "ttft_ms, tokens_per_s, peak_ram_mib, peak_vram_mib, cold_start_ms, api_ok, offline_ok, "
            "score_speed, score_balanced, score_accuracy, detail, created_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (d["id"], d["candidate_id"], d["task"], d["eval_version"], d["backend"],
             d["runner_version"], d["hardware_fingerprint"], _dumps(d["params"]), d["quality"],
             d["passed_cases"], d["total_cases"], d["median_latency_ms"], d["ttft_ms"],
             d["tokens_per_s"], d["peak_ram_mib"], d["peak_vram_mib"], d["cold_start_ms"],
             int(d["api_ok"]), int(d["offline_ok"]), d["score_speed"], d["score_balanced"],
             d["score_accuracy"], _dumps(d["detail"]), d["created_at"]))

    def list_evaluations(self, candidate_id: str | None = None,
                         eval_version: str | None = None) -> list[EvaluationRecord]:
        sql = "SELECT * FROM evaluations WHERE 1=1"
        params: list[Any] = []
        if candidate_id:
            sql += " AND candidate_id=?"
            params.append(candidate_id)
        if eval_version:
            sql += " AND eval_version=?"
            params.append(eval_version)
        sql += " ORDER BY created_at DESC"
        out = []
        for row in self.query(sql, params):
            data = dict(row)
            data["params"] = _loads(data["params"], {})
            data["detail"] = _loads(data["detail"], {})
            data["api_ok"] = bool(data["api_ok"])
            data["offline_ok"] = bool(data["offline_ok"])
            out.append(EvaluationRecord(**data))
        return out

    # ---------- 种子 ----------
    @staticmethod
    def _seed_from_row(row: dict) -> Seed:
        data = dict(row)
        data["manifest"] = _loads(data["manifest"], {})
        data["offline_ready"] = bool(data["offline_ready"])
        return Seed(**data)

    def put_seed(self, seed: Seed) -> None:
        d = seed.model_dump()
        self.execute(
            "INSERT INTO seeds(id, candidate_id, rel_dir, manifest, completeness, offline_ready, "
            "integrity, created_at, updated_at) VALUES(?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET manifest=excluded.manifest, "
            "completeness=excluded.completeness, offline_ready=excluded.offline_ready, "
            "integrity=excluded.integrity, updated_at=excluded.updated_at",
            (d["id"], d["candidate_id"], d["rel_dir"], _dumps(d["manifest"]), d["completeness"],
             int(d["offline_ready"]), d["integrity"], d["created_at"], d["updated_at"]))

    def get_seed(self, seed_id: str) -> Seed | None:
        row = self.query_one("SELECT * FROM seeds WHERE id=?", (seed_id,))
        return self._seed_from_row(row) if row else None

    def list_seeds(self) -> list[Seed]:
        return [self._seed_from_row(r) for r in self.query("SELECT * FROM seeds ORDER BY created_at DESC")]

    def find_seed_by_candidate(self, candidate_id: str) -> Seed | None:
        row = self.query_one("SELECT * FROM seeds WHERE candidate_id=? ORDER BY created_at DESC LIMIT 1",
                             (candidate_id,))
        return self._seed_from_row(row) if row else None

    # ---------- 默认版本 ----------
    def get_default(self, task: str, profile: str) -> DefaultPointer:
        row = self.query_one("SELECT * FROM defaults WHERE task=? AND profile=?", (task, profile))
        if not row:
            return DefaultPointer(task=task, profile=profile)
        return DefaultPointer(task=row["task"], profile=row["profile"], seed_id=row["seed_id"],
                              pinned=bool(row["pinned"]), previous_seed_id=row["previous_seed_id"],
                              source=row["source"], updated_at=row["updated_at"])

    def set_default(self, pointer: DefaultPointer) -> None:
        pointer.updated_at = utc_now()
        self.execute(
            "INSERT INTO defaults(task, profile, seed_id, pinned, previous_seed_id, source, updated_at) "
            "VALUES(?,?,?,?,?,?,?) ON CONFLICT(task, profile) DO UPDATE SET "
            "seed_id=excluded.seed_id, pinned=excluded.pinned, "
            "previous_seed_id=excluded.previous_seed_id, source=excluded.source, "
            "updated_at=excluded.updated_at",
            (pointer.task, pointer.profile, pointer.seed_id, int(pointer.pinned),
             pointer.previous_seed_id, pointer.source, pointer.updated_at))

    def list_defaults(self) -> list[DefaultPointer]:
        return [self.get_default(r["task"], r["profile"])
                for r in self.query("SELECT task, profile FROM defaults ORDER BY task, profile")]

    # ---------- 同步记录 / API key ----------
    def start_sync(self, report: SyncReport) -> None:
        self.execute("INSERT INTO sync_runs(id, started_at, status, trigger, detail) VALUES(?,?,?,?,?)",
                     (report.id, report.started_at, report.status, report.trigger,
                      _dumps(report.model_dump(exclude={"id", "started_at", "status", "trigger"}))))

    def finish_sync(self, report: SyncReport) -> None:
        self.execute("UPDATE sync_runs SET finished_at=?, status=?, detail=? WHERE id=?",
                     (report.finished_at, report.status,
                      _dumps(report.model_dump(exclude={"id", "started_at", "finished_at",
                                                        "status", "trigger"})), report.id))

    def list_sync_runs(self, limit: int = 20) -> list[dict]:
        rows = self.query("SELECT * FROM sync_runs ORDER BY started_at DESC LIMIT ?", (limit,))
        for r in rows:
            r["detail"] = _loads(r["detail"], {})
        return rows

    def add_api_key(self, key_hash: str, label: str) -> None:
        self.execute("INSERT OR IGNORE INTO api_keys(key_hash, label, created_at) VALUES(?,?,?)",
                     (key_hash, label, utc_now()))

    def has_api_key(self, key_hash: str) -> bool:
        return self.query_one("SELECT 1 AS ok FROM api_keys WHERE key_hash=?", (key_hash,)) is not None
