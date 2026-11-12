"""能力目录、三档选法与版本化评测配置（文档 §4）。

阶段 1 只启用通用对话；其余能力按文档 §4 表格登记，标明阶段与「待接入」，
由后续阶段接入各自的运行器和评测，不虚构能力已可用。
「入围和晋升分开」：页面标预估，默认档位只使用本机成绩。
"""
from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import EVAL_VERSION, FIXED_CONCURRENCY, FIXED_CONTEXT, PROMOTION, Paths
from .models import Candidate, EvaluationRecord, PROFILE_LABELS, PROFILES

# §4 能力按用户任务分类；同一个包可以服务多项能力和多个档位。
TASKS: dict[str, dict[str, Any]] = {
    "chat": {
        "label": "通用对话", "phase": 1, "enabled": True,
        "description": "通用语言、中文、长文与结构化输出",
        "metrics": ["通过率", "首 token 时间", "生成速度"],
        "trial": "流式聊天，可附加文本文件",
        "evaluation": "固定中英指令与事实提取样本；长上下文另设配置（阶段 2）",
        "config": "stage1",
    },
    "code": {
        "label": "专长语言 · 代码", "phase": 2, "enabled": False,
        "description": "代码任务通过率，先做可判定输出",
        "metrics": ["可判定通过率"], "trial": "代码任务",
        "evaluation": "独立测试集；执行模型产出的代码须另做隔离环境",
        "config": None,
    },
    "math": {
        "label": "专长语言 · 数学", "phase": 3, "enabled": False,
        "description": "数学可判定答案", "metrics": ["答案正确率"],
        "trial": "数学问答", "evaluation": "独立测试集", "config": None,
    },
    "translation": {
        "label": "专长语言 · 翻译", "phase": 2, "enabled": False,
        "description": "翻译固定样例", "metrics": ["固定样例评分"],
        "trial": "翻译", "evaluation": "固定样例", "config": None,
    },
    "vision": {
        "label": "图片理解", "phase": 2, "enabled": False,
        "description": "看图问答、分类、检测与分割；通用看图与检测分割分别排名",
        "metrics": ["看图问答正确率", "mAP", "IoU"], "trial": "上传本地图片",
        "evaluation": "固定图片集", "config": None,
    },
    "ocr": {
        "label": "文字识别", "phase": 2, "enabled": False,
        "description": "OCR 与版面提取", "metrics": ["CER", "字段提取正确率"],
        "trial": "上传截图或扫描件，返回文字与坐标", "evaluation": "带标注中英图片",
        "config": None,
    },
    "image_gen": {
        "label": "图片生成与编辑", "phase": 3, "enabled": False,
        "description": "固定提示、种子、分辨率与步数", "metrics": ["完成时间", "峰值显存", "人工盲评"],
        "trial": "文生图 / 图生图", "evaluation": "固定提示集", "config": None,
    },
    "speech": {
        "label": "语音转写", "phase": 3, "enabled": False,
        "description": "ASR 的 CER/WER 与实时因子；TTS 可懂度",
        "metrics": ["CER/WER", "实时因子"], "trial": "录音转写 / 文本朗读",
        "evaluation": "固定音频集", "config": None,
    },
    "video": {
        "label": "视频理解", "phase": 4, "enabled": False,
        "description": "理解与生成分开显示硬件门槛",
        "metrics": ["时间定位", "跨帧一致性"], "trial": "短片问答 / 文生视频",
        "evaluation": "固定片段与提示", "config": None,
    },
    "retrieval": {
        "label": "嵌入、重排与本地检索", "phase": 4, "enabled": False,
        "description": "固定问题与本地文档", "metrics": ["Recall@k", "nDCG"],
        "trial": "本地文档检索", "evaluation": "固定文档集", "config": None,
    },
    "pose": {"label": "人体姿态", "phase": 2, "enabled": True},
    "hand": {"label": "手部关节", "phase": 2, "enabled": True},
    "face_swap": {"label": "换脸", "phase": 3, "enabled": False},
    "image_edit": {"label": "图片编辑", "phase": 3, "enabled": False},
    "tts": {"label": "语音合成", "phase": 3, "enabled": False},
    "video_gen": {"label": "视频生成", "phase": 4, "enabled": False},
}

TASK_GROUPS = {
    "语言": ("chat", "code", "math", "translation", "retrieval"),
    "图像": ("vision", "ocr", "image_gen", "image_edit", "face_swap"),
    "动作": ("pose", "hand"),
    "音频": ("speech", "tts"),
    "视频": ("video", "video_gen"),
}

TASK_INPUTS = {
    "chat": ["messages"], "code": ["messages"], "math": ["messages"],
    "translation": ["messages"], "retrieval": ["text"],
    "vision": ["image", "text"], "ocr": ["image"], "image_gen": ["text"],
    "face_swap": ["source_image", "target_image"], "pose": ["image"],
    "hand": ["image"], "speech": ["audio"], "video": ["video", "text"],
    "image_edit": ["image", "text"], "tts": ["text"], "video_gen": ["text"],
}


def sota_leaders(candidates: list[Candidate]) -> dict[str, list[dict]]:
    """最高分只在同任务、基准协议、指标与方向内比较。"""
    groups: dict[tuple, list[tuple[Candidate, Any]]] = {}
    for c in candidates:
        for e in c.evidence:
            if (e.benchmark_key and e.metric and e.url and e.variant == c.variant
                    and e.model_version == c.commit_rev and e.value is not None
                    and math.isfinite(e.value)):
                groups.setdefault((c.task, e.benchmark_key, e.metric, e.higher_is_better), []).append((c, e))
    leaders: dict[str, list[dict]] = {}
    for (_, _, _, higher), entries in groups.items():
        best = (max if higher else min)(e.value for _, e in entries)
        count = len({c.id for c, _ in entries})
        for c, e in entries:
            if e.value == best:
                leaders.setdefault(c.id, []).append({**e.model_dump(), "compared": count})
    return leaders

PROFILE_DEFS = {
    "speed": {"label": PROFILE_LABELS["speed"], "rule": "先过最低质量与稳定性门槛，再按端到端时间排序；并列时看质量",
              "extras": "同时展示首 token 时间与 tokens/s"},
    "balanced": {"label": PROFILE_LABELS["balanced"], "rule": "0.6 × Q + 0.4 × S，S = min(1, 目标耗时 / 实测中位耗时)",
                 "extras": "没有量化质量信号的任务先做人工对比，暂不开自动替换"},
    "accuracy": {"label": PROFILE_LABELS["accuracy"], "rule": "稳定完成且不超过最大等待时间者中选质量最高；同分看时间",
                 "extras": "量化位数只是变体信息"},
}

# §5.2 的「受控候选」：Agent 连接失败时定时任务仍可刷新目录并使用受控候选。
# 两条来自文档 §3 的第一轮启动候选，尚未实测，不预先认定任何档位冠军。
CONTROLLED_CANDIDATES: list[dict[str, Any]] = [
    {
        "source": "huggingface", "repo": "ggml-org/Qwen3.5-0.8B-GGUF", "commit_rev": "main",
        "variant": "Qwen3.5-0.8B-Q4_0.gguf", "task": "chat", "publisher": "ggml-org",
        "license": "apache-2.0", "backend": "cpu", "size_bytes": 0,
        "files": ["Qwen3.5-0.8B-Q4_0.gguf", "README.md", "LICENSE"],
        "rationale": "文档 §3 指定的 CPU 应急种子启动候选，固定 2048 token 上下文、并发 1；未实测",
        "evidence": [{"title": "上游 GGUF 仓库", "url": "https://huggingface.co/ggml-org/Qwen3.5-0.8B-GGUF",
                      "source": "Hugging Face 模型卡", "note": "原始精度成绩不得挪给量化版"}],
    },
    {
        "source": "huggingface", "repo": "Qwen/Qwen3-4B-GGUF", "commit_rev": "main",
        "variant": "Qwen3-4B-Q4_K_M.gguf", "task": "chat", "publisher": "Qwen",
        "license": "apache-2.0", "backend": "cuda", "size_bytes": 0,
        "files": ["Qwen3-4B-Q4_K_M.gguf", "README.md", "LICENSE"],
        "rationale": "文档 §3 指定的独显链路启动候选；能否纳入档位取决于本机实测",
        "evidence": [{"title": "上游 GGUF 仓库", "url": "https://huggingface.co/Qwen/Qwen3-4B-GGUF/tree/main",
                      "source": "Hugging Face 模型卡", "note": "其他机器的速度只作参考"}],
    },
]


@lru_cache(maxsize=8)
def load_eval_config(name: str = "stage1") -> dict[str, Any]:
    path = Path(__file__).parent / "resources" / "eval" / f"{name}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("version", EVAL_VERSION)
    data.setdefault("context", FIXED_CONTEXT)
    data.setdefault("concurrency", FIXED_CONCURRENCY)
    return data


def load_debug_samples() -> list[dict[str, Any]]:
    """§5.1 Agent 只能用单独的调试样例，不能接触正式题集。"""
    path = Path(__file__).parent / "resources" / "eval" / "debug_samples.json"
    return json.loads(path.read_text(encoding="utf-8"))["cases"]


def grade_case(case: dict[str, Any], output: str) -> tuple[bool, str]:
    """固定输入、期望输出与评分规则由项目版本管理。"""
    expect = case.get("expect", {})
    kind = expect.get("type", "contains_any")
    text = (output or "").strip()
    if kind == "contains_any":
        values = expect.get("values", [])
        ok = any(v in text for v in values)
        return ok, f"期望包含任一 {values}；实际前 80 字：{text[:80]}"
    if kind == "contains_all":
        values = expect.get("values", [])
        ok = all(v in text for v in values)
        return ok, f"期望全部包含 {values}；实际前 80 字：{text[:80]}"
    if kind == "regex":
        ok = re.search(expect.get("pattern", ""), text, re.S) is not None
        return ok, f"期望匹配 {expect.get('pattern')}；实际前 80 字：{text[:80]}"
    if kind == "exact":
        ok = text == expect.get("value", "")
        return ok, f"期望精确等于 {expect.get('value')!r}；实际 {text[:80]!r}"
    if kind == "json_format":
        import json as _json
        try:
            obj = _json.loads(text)
        except (ValueError, TypeError):
            return False, f"输出不是合法 JSON：{text[:80]}"
        keys = expect.get("keys", [])
        ok = all(k in obj for k in keys)
        return ok, f"期望 JSON 键 {keys}；实际 {str(obj)[:80]}"
    return False, f"未知评分规则 {kind}"


def compute_scores(record: EvaluationRecord, config: dict[str, Any]) -> tuple[float, float, float]:
    """返回 (速度分, 平衡分, 精度分)。"""
    q = record.quality
    latency = max(1.0, record.median_latency_ms)
    target = float(config.get("target_latency_ms", 6000))
    s = min(1.0, target / latency)
    speed = 1000.0 / latency  # 单调等价于「端到端时间越短越好」
    balanced = (PROMOTION["balanced_quality_weight"] * q
                + PROMOTION["balanced_speed_weight"] * s)
    return round(speed, 6), round(balanced, 6), q


def meets_gates(record: EvaluationRecord, config: dict[str, Any]) -> tuple[bool, str]:
    """默认档位只使用本机成绩；API 通过、断网可用和评分过线分别记录。"""
    if record.eval_version != config["version"]:
        return False, f"评测版本 {record.eval_version} 与当前 {config['version']} 不一致，需按新口径重测"
    if not record.api_ok:
        return False, "正式 API 复测未通过"
    if not record.offline_ok:
        return False, "断网启动验收未通过"
    gates = config.get("gates", {})
    if record.quality < gates.get("min_quality", 0.7):
        return False, f"质量分 {record.quality:.2f} 低于门槛 {gates.get('min_quality')}"
    if record.passed_cases < gates.get("min_passed_cases", 1):
        return False, f"通过用例数 {record.passed_cases} 低于门槛"
    if record.median_latency_ms > gates.get("max_wait_ms", 30000):
        return False, f"中位耗时 {record.median_latency_ms:.0f} ms 超过最大等待时间"
    if not record.detail.get("stable", False):
        return False, "重复运行不稳定"
    return True, "通过质量、稳定性、等待时间与离线门槛"


def promote(current: EvaluationRecord | None, challenger: EvaluationRecord,
            profile: str, config: dict[str, Any]) -> tuple[bool, str]:
    """§4 换档门槛：速度档快 10%、精度档多通过 2 个样本、平衡分高 0.03。"""
    ok, reason = meets_gates(challenger, config)
    if not ok:
        return False, reason
    if current is None or not meets_gates(current, config)[0]:
        return True, "当前档位无有效版本，候选通过全部门槛后设为默认"
    if profile == "speed":
        gain = (current.median_latency_ms - challenger.median_latency_ms) / max(1.0, current.median_latency_ms)
        if gain < PROMOTION["speed_min_relative_gain"]:
            return False, f"速度提升 {gain:.1%} 未达 {PROMOTION['speed_min_relative_gain']:.0%} 门槛"
        return True, f"端到端时间快 {gain:.1%}"
    if profile == "accuracy":
        extra = challenger.passed_cases - current.passed_cases
        if extra < PROMOTION["accuracy_extra_passed_cases"] and challenger.quality <= current.quality:
            return False, f"质量未提高（多通过 {extra} 个样本，门槛 {PROMOTION['accuracy_extra_passed_cases']}）"
        return True, f"质量分 {challenger.quality:.2f}，多通过 {extra} 个样本"
    delta = challenger.score_balanced - current.score_balanced
    if delta < PROMOTION["balanced_min_delta"]:
        return False, f"平衡分提升 {delta:.3f} 未达 {PROMOTION['balanced_min_delta']} 门槛"
    return True, f"平衡分提升 {delta:.3f}"


def rank(records: list[EvaluationRecord], profile: str) -> list[EvaluationRecord]:
    key = {"speed": "score_speed", "balanced": "score_balanced",
           "accuracy": "score_accuracy"}.get(profile, "score_balanced")
    return sorted(records, key=lambda r: (-getattr(r, key), r.median_latency_ms))


def task_view() -> list[dict[str, Any]]:
    out = []
    for key, item in TASKS.items():
        out.append({
            "id": key, **item,
            "group": next(name for name, tasks in TASK_GROUPS.items() if key in tasks),
            "inputs": TASK_INPUTS[key],
            "profiles": [{"id": p, "label": PROFILE_LABELS[p]} for p in PROFILES],
        })
    return out


def profile_view() -> list[dict[str, Any]]:
    return [{"id": p, **PROFILE_DEFS[p]} for p in PROFILES]
