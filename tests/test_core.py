"""§3、§4 的规则单测：容量预算、评分口径、换档门槛与评测配置。"""
from __future__ import annotations

from doomsday_seed.catalog import (CONTROLLED_CANDIDATES, TASKS, compute_scores,
                                   grade_case, load_debug_samples, load_eval_config,
                                   promote, rank)
from doomsday_seed.config import GIB, MIB, PROMOTION, ram_reserve, vram_reserve
from doomsday_seed.hardware import check_capacity, estimate_footprint
from doomsday_seed.models import Candidate, EvaluationRecord, HardwareSample


def _record(**kwargs) -> EvaluationRecord:
    base = dict(
        id="e1", candidate_id="c1", eval_version="stage1-1.0.0", backend="cpu",
        runner_version="b-test", hardware_fingerprint="fp", quality=0.9,
        passed_cases=7, total_cases=8, median_latency_ms=1000.0,
        score_balanced=0.9, score_speed=1.0, score_accuracy=0.9,
        api_ok=True, offline_ok=True,
        detail={"stable": True},
    )
    base.update(kwargs)
    return EvaluationRecord(**base)


def test_reserve_rules():
    assert ram_reserve(8 * GIB) == 2 * GIB
    assert ram_reserve(32 * GIB) == 3.2 * GIB
    assert vram_reserve(4 * GIB) == 1 * GIB
    assert vram_reserve(20 * GIB) == 2 * GIB


def test_footprint_and_capacity():
    footprint = estimate_footprint(500 * MIB, context=2048, gpu_layers=0, backend="cpu")
    assert footprint["est_ram_mib"] > 500
    assert footprint["est_vram_mib"] == 0
    sample = HardwareSample(ram_total_mib=32 * 1024, ram_available_mib=2 * 1024,
                            vram_total_mib=8 * 1024, vram_available_mib=7 * 1024)
    # 可用 RAM 2 GiB，预留 3.2 GiB → 余量为 0，必然不通过且给出可读原因。
    result = check_capacity(footprint, sample, "cpu")
    assert result["fits"] is False
    assert "内存余量不足" in result["reason"]
    assert result["budget"]["ram_reserve_mib"] == 3.2 * 1024


def test_balanced_score_formula():
    config = load_eval_config()
    record = _record(quality=0.8, median_latency_ms=3000.0)
    speed, balanced, accuracy = compute_scores(record, config)
    expected_s = min(1.0, config["target_latency_ms"] / 3000.0)
    assert balanced == round(PROMOTION["balanced_quality_weight"] * 0.8
                             + PROMOTION["balanced_speed_weight"] * expected_s, 6)
    assert speed == round(1000 / 3000, 6)
    assert accuracy == 0.8


def test_promotion_thresholds():
    config = load_eval_config()
    current = _record(id="current", median_latency_ms=1000.0, score_balanced=0.90,
                      passed_cases=7, quality=0.875)
    # 速度档：快 9% 不换，快 12% 换。
    slow = _record(id="slow", median_latency_ms=910.0)
    assert promote(current, slow, "speed", config)[0] is False
    fast = _record(id="fast", median_latency_ms=880.0)
    assert promote(current, fast, "speed", config)[0] is True
    # 平衡档：+0.02 不换，+0.04 换。
    assert promote(current, _record(id="b1", score_balanced=0.92), "balanced", config)[0] is False
    assert promote(current, _record(id="b2", score_balanced=0.94), "balanced", config)[0] is True
    # 精度档：多通过 1 个样本且质量未提高 → 不换；多通过 2 个 → 换。
    assert promote(current, _record(id="a1", passed_cases=8, quality=0.875),
                   "accuracy", config)[0] is False
    assert promote(current, _record(id="a2", passed_cases=9, quality=1.0),
                   "accuracy", config)[0] is True


def test_gates_are_separate_from_promotion():
    config = load_eval_config()
    assert promote(None, _record(), "balanced", config)[0] is True
    assert promote(None, _record(api_ok=False), "balanced", config)[0] is False
    assert promote(None, _record(offline_ok=False), "balanced", config)[0] is False
    assert promote(None, _record(detail={"stable": False}), "balanced", config)[0] is False
    assert promote(None, _record(eval_version="old"), "balanced", config)[0] is False


def test_eval_config_is_versioned_and_separate_from_debug_samples():
    config = load_eval_config()
    debug = load_debug_samples()
    assert config["version"] == "stage1-1.0.0"
    assert config["protocol"]["context"] == 2048
    assert config["protocol"]["concurrency"] == 1
    assert len(config["cases"]) >= 8
    assert {c["id"] for c in config["cases"]}.isdisjoint({c["id"] for c in debug})
    assert len(rank([_record(id="a", score_balanced=0.7),
                     _record(id="b", score_balanced=0.95)], "balanced")) == 2


def test_grade_case_rules():
    assert grade_case({"expect": {"type": "contains_any", "values": ["北京"]}}, "北京")[0]
    assert not grade_case({"expect": {"type": "contains_all", "values": ["甲", "乙"]}}, "甲")[0]
    assert grade_case({"expect": {"type": "regex", "pattern": r"A-\d+"}}, "A-17")[0]
    assert grade_case({"expect": {"type": "json_format", "keys": ["ok"]}}, '{"ok": 1}')[0]
    assert grade_case({"expect": {"type": "exact", "value": "OK"}}, "OK")[0]


def test_idempotency_key_and_task_phases():
    c = Candidate(id=Candidate.make_id("huggingface", "org/repo", "abc", "Q4_K_M.gguf"),
                  repo="org/repo", commit_rev="abc", variant="Q4_K_M.gguf")
    assert c.idempotency_key == "huggingface|org/repo|abc|Q4_K_M.gguf"
    assert {t for t, v in TASKS.items() if v["enabled"]} == {"chat", "pose", "hand"}
    assert TASKS["ocr"]["phase"] == 2 and TASKS["video"]["phase"] == 4
    assert {c["repo"] for c in CONTROLLED_CANDIDATES} == {
        "ggml-org/Qwen3.5-0.8B-GGUF", "Qwen/Qwen3-4B-GGUF"}
