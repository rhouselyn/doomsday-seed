"""领域对象与请求/响应模型。

§5：目录与作业分开记录；`来源 + 仓库 + commit + 变体` 是下载幂等键。
所有对象都可直接 model_dump() 落 SQLite / 通过 API 返回。
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# §5 目录状态：候选 / 已下载·待验证 / 离线可用 / 暂不支持 / 待授权
CANDIDATE_STATES = ("candidate", "downloaded", "offline_ready", "unsupported", "gated")
STATE_LABELS = {
    "candidate": "候选",
    "downloaded": "已下载 · 待验证",
    "offline_ready": "离线可用",
    "unsupported": "暂不支持",
    "gated": "待授权",
}
STATE_ICONS = {
    "candidate": "○",
    "downloaded": "◐",
    "offline_ready": "✓",
    "unsupported": "×",
    "gated": "🔒",
}

# §5 作业：排队 → 下载 → 校验 → 调试 → 复测 → 完成；异常进入等待网络/等待资源/失败
JOB_STATES = (
    "queued", "downloading", "verifying", "debugging", "retesting", "done",
    "waiting_network", "waiting_resource", "failed",
)
JOB_STAGES = ("排队", "下载", "校验", "调试", "复测", "完成")
JOB_STATE_LABELS = {
    "queued": "排队", "downloading": "下载", "verifying": "校验",
    "debugging": "调试", "retesting": "复测", "done": "完成",
    "waiting_network": "等待网络", "waiting_resource": "等待资源", "failed": "失败",
}

PROFILES = ("speed", "balanced", "accuracy")
PROFILE_LABELS = {"speed": "速度优先", "balanced": "平衡", "accuracy": "精度优先"}


def utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class FileSpec(BaseModel):
    """固定文件清单中的一项；路径一律为包内相对路径。"""

    repo_path: str
    local_path: str
    size_bytes: int = 0
    kind: Literal["weight", "projection", "config", "license", "model_card", "usage", "sample"] = "weight"
    upstream_sha256: str | None = None
    sha256: str | None = None


class RuntimeConfig(BaseModel):
    """§5.1 由 Agent 在允许范围内调通、再由脚本冻结的运行配置。"""

    adapter: str = "llamacpp-openai"
    backend: Literal["cpu", "cuda"] = "cpu"
    runner_version: str = "unregistered"
    gpu_layers: int = 0
    context: int = 2048
    concurrency: int = 1
    chat_template: str | None = None
    mmproj: str | None = None
    extra_args: list[str] = Field(default_factory=list)
    verified: bool = False

    def fingerprint(self) -> str:
        import hashlib
        import json
        payload = self.model_dump(exclude={"verified"})
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


class Evidence(BaseModel):
    """公开证据：必须附来源、模型版本、测试集与协议。"""

    title: str
    url: str = ""
    source: str = ""
    model_version: str = ""
    dataset: str = ""
    protocol: str = ""
    note: str = ""
    benchmark_key: str = ""
    metric: str = ""
    value: float | None = None
    higher_is_better: bool = True
    variant: str = ""


class Estimate(BaseModel):
    """兼容性预估（页面标「预估」，不是实测成绩）。"""

    backend: str = "cpu"
    est_ram_mib: float = 0
    est_vram_mib: float = 0
    basis: str = ""
    fits: bool | None = None
    reason: str = ""


class Candidate(BaseModel):
    id: str
    source: str = "huggingface"
    repo: str
    commit_rev: str
    variant: str
    task: str = "chat"
    publisher: str = ""
    license: str = ""
    size_bytes: int = 0
    files: list[FileSpec] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    estimate: Estimate = Field(default_factory=Estimate)
    rationale: str = ""
    state: str = "candidate"
    reason: str = ""
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)

    @staticmethod
    def make_id(source: str, repo: str, commit: str, variant: str) -> str:
        """§5 下载幂等键：来源 + 仓库 + commit + 变体。"""
        import hashlib
        raw = f"{source}|{repo}|{commit}|{variant}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:24]

    @property
    def idempotency_key(self) -> str:
        return f"{self.source}|{self.repo}|{self.commit_rev}|{self.variant}"


class DownloadPlan(BaseModel):
    """§5「按固定 commit 和精确文件清单先做下载预检」。"""

    candidate_id: str
    repo: str
    commit_rev: str
    files: list[FileSpec]
    runtime_config: RuntimeConfig = Field(default_factory=RuntimeConfig)
    total_bytes: int = 0
    preflight: dict[str, Any] = Field(default_factory=dict)


class AgentSuggestion(BaseModel):
    """§5 Agent 输出的结构化下载建议。"""

    repo: str
    commit_rev: str
    variant: str
    task: str = "chat"
    profile: Literal["speed", "balanced", "accuracy"] | None = None
    files: list[str] = Field(default_factory=list)
    backend: str = "cpu"
    estimate_basis: str = ""
    rationale: str = ""
    evidence: list[Evidence] = Field(default_factory=list)
    controlled: bool = False  # Agent 不可用时使用的受控候选


class Job(BaseModel):
    id: str
    kind: str = "install"
    candidate_id: str = ""
    idempotency_key: str = ""
    state: str = "queued"
    stage: str = "排队"
    progress: float = 0.0
    message: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)


class EvaluationRecord(BaseModel):
    """§4「以硬件与驱动 + 运行器版本 + 模型变体 + 参数关联结果」。"""

    id: str
    candidate_id: str
    task: str = "chat"
    eval_version: str = "stage1-1.0.0"
    backend: str = "cpu"
    runner_version: str = "unregistered"
    hardware_fingerprint: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    quality: float = 0.0
    passed_cases: int = 0
    total_cases: int = 0
    median_latency_ms: float = 0.0
    ttft_ms: float = 0.0
    tokens_per_s: float = 0.0
    peak_ram_mib: float = 0.0
    peak_vram_mib: float = 0.0
    cold_start_ms: float = 0.0
    api_ok: bool = False
    offline_ok: bool = False
    score_speed: float = 0.0
    score_balanced: float = 0.0
    score_accuracy: float = 0.0
    detail: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now)


class Seed(BaseModel):
    id: str
    candidate_id: str
    rel_dir: str = ""
    manifest: dict[str, Any] = Field(default_factory=dict)
    completeness: float = 0.0
    offline_ready: bool = False
    integrity: str = "unchecked"
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)


class DefaultPointer(BaseModel):
    task: str
    profile: str
    seed_id: str | None = None
    pinned: bool = False
    previous_seed_id: str | None = None
    source: str = "auto"
    updated_at: str = Field(default_factory=utc_now)


class Settings(BaseModel):
    """与 §6 `GET/PUT /api/settings` 对应。修改后返回实际生效状态。"""

    auto_update: bool = True
    schedule_enabled: bool = True
    schedule_time: str = "03:00"
    agent_enabled: bool = True
    # 可选：兼容 OpenAI Chat Completions 的模型服务；留空使用内置默认。
    agent_base_url: str = ""
    agent_model: str = ""
    # 归档根目录；留空使用 data/seeds。修改时现有模型会一起移动。
    download_root: str = ""
    trusted_publishers: list[str] = Field(default_factory=list)
    manual_repos: list[str] = Field(default_factory=list)
    subscriptions: dict[str, list[str]] = Field(default_factory=lambda: {"chat": ["balanced"]})
    download_modes: dict[str, Literal["local", "sota", "off"]] = Field(default_factory=dict)
    sota_subscriptions: list[str] = Field(default_factory=list)
    simulated_offline: bool = False  # 仅自查；真实断网演练须系统层切断外网

    def active_pairs(self) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        for task, profiles in self.subscriptions.items():
            if self.download_modes.get(task, "local") != "local":
                continue
            for profile in profiles:
                if profile in PROFILES:
                    pairs.append((task, profile))
        return pairs


class SyncReport(BaseModel):
    id: str
    trigger: str = "manual"
    status: str = "running"
    started_at: str = Field(default_factory=utc_now)
    finished_at: str | None = None
    discovered: int = 0
    shortlisted: list[str] = Field(default_factory=list)
    jobs: list[str] = Field(default_factory=list)
    skipped: list[dict[str, str]] = Field(default_factory=list)
    message: str = ""


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = "user"
    content: str = ""


class ChatCompletionRequest(BaseModel):
    model: str = "chat:balanced"
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None


class HardwareSample(BaseModel):
    sampled_at: str = Field(default_factory=utc_now)
    os: str = ""
    cpu_model: str = ""
    arch: str = ""
    instruction_sets: list[str] = Field(default_factory=list)
    instruction_source: str = "unknown"
    cores: int = 0
    threads: int = 0
    ram_total_mib: float = 0.0
    ram_available_mib: float = 0.0
    gpus: list[dict[str, Any]] = Field(default_factory=list)
    vram_total_mib: float = 0.0
    vram_available_mib: float = 0.0
    driver_version: str = ""
    backend_available: dict[str, bool] = Field(default_factory=dict)
    backend_detail: dict[str, str] = Field(default_factory=dict)
    runner_version: str = "unregistered"

    def fingerprint(self) -> str:
        """§3 结果以「硬件与驱动 + 运行器版本」关联，不跨机器沿用。"""
        import hashlib
        raw = "|".join([
            self.cpu_model, self.arch, ",".join(sorted(self.instruction_sets)),
            str(int(self.ram_total_mib)),
            ",".join(sorted(g.get("name", "") for g in self.gpus)),
            self.driver_version, self.runner_version,
        ])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
