"""全局常量与路径。

对应文档 §2「单机结构与职责」：数据库保存相对路径，归档与程序源码分开；
§3 的容量预留规则、§4 的晋升门槛、§5 的下载上限都集中在此，便于版本化。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

APP_NAME = "末日种子"
APP_TAGLINE = "为断网之后，保存仍能运行的 AI。"
APP_VERSION = "0.3.0"

# §4 评测配置是版本化的；改口径后旧成绩必须重测，不能沿用。
EVAL_VERSION = "stage1-1.0.0"
# §3 首轮固定场景：2048 token 上下文、并发 1。
FIXED_CONTEXT = 2048
FIXED_CONCURRENCY = 1

LOOPBACK_HOST = "127.0.0.1"
DEFAULT_PORT = 8777
GIB = 1024 ** 3
MIB = 1024 ** 2

# §3「先做容量预估」：可调整的起始规则。
def ram_reserve(total_bytes: float) -> float:
    return max(2 * GIB, total_bytes * 0.10)


def vram_reserve(total_bytes: float) -> float:
    return max(1 * GIB, total_bytes * 0.10)


# §4「入围和晋升分开」：首版换档门槛。
PROMOTION = {
    "speed_min_relative_gain": 0.10,
    "accuracy_extra_passed_cases": 2,
    "balanced_min_delta": 0.03,
    "balanced_quality_weight": 0.6,
    "balanced_speed_weight": 0.4,
}

# §5「限量下载」：每轮每项能力、每个订阅档位最多自动试装一个新变体。
MAX_NEW_VARIANTS_PER_TASK_PROFILE_PER_RUN = 1
# §5「单源出错保留旧结果」「429 遵从平台限流规则，带退避和重试上限」。
HF_MAX_RETRIES = 4
HF_MAX_WORKERS = 5
HF_BACKOFF_BASE_SECONDS = 1.5
HF_CURSOR_OVERLAP = 20
# §6 上传限制（首版聊天可附带一个文本文件）。
MAX_UPLOAD_BYTES = 2 * MIB
ALLOWED_TEXT_SUFFIXES = (".txt", ".md", ".json", ".csv", ".log")

DEFAULT_SCHEDULE_TIME = "03:00"
DEFAULT_TRUSTED_PUBLISHERS = ("Qwen", "ggml-org", "unsloth", "bartowski", "lmstudio-community")


@dataclass
class Paths:
    """所有落盘位置。data 可通过 DOOMSDAY_DATA 覆盖（离线恢复/测试用）。"""

    root: Path
    data: Path
    # 归档根目录可在配置页修改；为空时使用 data/seeds。
    download_root: str = ""
    _seeds_root: Path | None = None
    _staging_root: Path | None = None

    @classmethod
    def detect(cls, data_dir: str | os.PathLike | None = None) -> "Paths":
        root = Path(__file__).resolve().parents[2]
        raw = data_dir or os.environ.get("DOOMSDAY_DATA") or (root / "data")
        return cls(root=root, data=Path(raw).resolve())

    # 程序与静态资源
    @property
    def package(self) -> Path:
        return self.root / "src" / "doomsday_seed"

    @property
    def web(self) -> Path:
        return self.root / "web"

    @property
    def web_assets(self) -> Path:
        return self.web / "assets"

    @property
    def eval_resources(self) -> Path:
        return self.package / "resources" / "eval"

    @property
    def fixtures(self) -> Path:
        return self.root / "tests" / "fixtures"

    # 数据与归档（与源码分开）
    @property
    def db(self) -> Path:
        return self.data / "seed-vault.sqlite3"

    @property
    def staging(self) -> Path:
        return self._staging_root or (self.data / "staging")

    @property
    def seeds(self) -> Path:
        return self._seeds_root or (self.data / "seeds")

    @property
    def default_seeds(self) -> Path:
        return self.data / "seeds"

    def resolve_download_root(self, raw: str | None) -> Path:
        """校验归档目录：必须是绝对路径，留空回到 data/seeds。"""
        text = (raw or "").strip()
        if not text:
            return self.default_seeds.resolve()
        target = Path(os.path.expandvars(text)).expanduser()
        if not target.is_absolute():
            raise ValueError("请填写绝对路径，例如 D:\\models\\seed-vault")
        return target.resolve()

    def set_download_root(self, raw: str | None = None) -> Path:
        """切换归档根目录；下载暂存区跟随同一卷，避免跨卷无法原子归档。"""
        target = self.resolve_download_root(raw)
        default = self.default_seeds.resolve()
        self.download_root = "" if target == default else str(target)
        self._seeds_root = None if target == default else target
        self._staging_root = (None if target == default else
                              target.parent / f"{target.name}-staging")
        self.seeds.mkdir(parents=True, exist_ok=True)
        self.staging.mkdir(parents=True, exist_ok=True)
        return self.seeds

    @property
    def runtimes(self) -> Path:
        return self.data / "runtimes"

    @property
    def runner_registry(self) -> Path:
        return self.runtimes / "registry.json"

    @property
    def exports(self) -> Path:
        return self.data / "exports"

    @property
    def session_token(self) -> Path:
        return self.data / "session.token"

    @property
    def api_key(self) -> Path:
        return self.data / "api-key"

    @property
    def sync_lock(self) -> Path:
        return self.data / "sync.lock"

    def ensure(self) -> None:
        for p in (self.data, self.staging, self.seeds, self.runtimes, self.exports, self.eval_resources):
            p.mkdir(parents=True, exist_ok=True)
