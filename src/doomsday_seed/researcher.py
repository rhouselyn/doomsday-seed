"""Read-only model discovery and its saved report."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from html import unescape
import json
import math
from pathlib import Path
import re

import httpx

from .agent import AgentContext, LocalRuleAgent
from .config import HF_MAX_WORKERS
from .models import utc_now

REVIEW_SOURCES = Path(__file__).with_name("resources") / "model-research.json"
ARENA_DATA = "https://datasets-server.huggingface.co/rows"
REVIEW_CACHE = "agent:public_scores"


def _arena_rows(config: str) -> list[dict]:
    response = httpx.get(ARENA_DATA, params={
        "dataset": "lmarena-ai/leaderboard-dataset", "config": config,
        "split": "latest", "offset": 0, "length": 100,
    }, timeout=12)
    response.raise_for_status()
    rows = [entry["row"] for entry in response.json()["rows"]]
    return sorted((row for row in rows if row.get("category") == "overall"
                   and row.get("license") and row["license"] != "Proprietary"
                   and isinstance(row.get("rating"), (int, float))
                   and math.isfinite(row["rating"])), key=lambda row: row["rank"])


def _aa_rows(url: str) -> list[dict]:
    # 只读取发布方榜单的首张表。表结构变化时由已有快照兜底。
    with httpx.stream("GET", url, timeout=12) as response:
        response.raise_for_status()
        prefix = bytearray()
        for block in response.iter_bytes():
            prefix.extend(block)
            if b"</table>" in prefix or len(prefix) > 256_000:
                break
    table = re.search(r"<table\b[^>]*>(.*?)</table>", prefix.decode("utf-8", errors="replace"), re.S)
    if not table:
        return []
    rows = []
    for row in re.findall(r"<tr\b[^>]*>(.*?)</tr>", table[1], re.S):
        cells = re.findall(r"<td\b[^>]*>(.*?)</td>", row, re.S)
        if len(cells) < 5:
            continue
        columns = [" ".join(unescape(re.sub(r"<[^>]+>", " ", cell)).split()) for cell in cells[:5]]
        number, _, name, score, votes = columns
        interval = re.match(r"(\d+)\s*±\s*(\d+)", score)
        if not number.isdigit() or not interval or not votes.replace(",", "").isdigit():
            continue
        value, error = map(int, interval.groups())
        rows.append({"model_name": name, "rank": int(number), "rating": value,
                     "rating_lower": value - error, "rating_upper": value + error,
                     "vote_count": int(votes.replace(",", "")), "read_date": utc_now()[:10]})
    return sorted(rows, key=lambda row: row["rank"])


def refresh_reviews(db) -> None:
    """刷新发布方开放数据。失败时保留最近一次成功的快照。"""
    candidates = json.loads(REVIEW_SOURCES.read_text(encoding="utf-8"))
    keys = sorted({e.get("arena_config") or e["url"] for c in candidates for e in c["evidence"]
                   if "arena_config" in e or "aa_name" in e})
    snapshot = json.loads(db.get_meta(REVIEW_CACHE, "{}"))

    def fetch(key):
        try:
            return _aa_rows(key) if key.startswith("https://artificialanalysis.ai/") else _arena_rows(key)
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            return []

    with ThreadPoolExecutor(max_workers=HF_MAX_WORKERS) as pool:
        for key, rows in zip(keys, pool.map(fetch, keys)):
            if rows:
                snapshot[key] = rows
    updated = json.dumps(snapshot, ensure_ascii=False, sort_keys=True)
    if updated != db.get_meta(REVIEW_CACHE):
        db.set_meta(REVIEW_CACHE, updated)


def model_reviews(db) -> list[dict]:
    """内置研究 Agent 按任务比较独立来源中的相对名次。"""
    candidates = json.loads(REVIEW_SOURCES.read_text(encoding="utf-8"))
    snapshot = json.loads(db.get_meta(REVIEW_CACHE, "{}"))
    by_task: dict[str, list[dict]] = {}
    for candidate in candidates:
        for evidence in candidate["evidence"]:
            key = evidence.get("arena_config") or evidence["url"]
            if key not in snapshot:
                continue
            name = evidence.get("aa_name") or candidate.get("arena_name")
            match = next((
                (rank, row) for rank, row in enumerate(snapshot[key], 1)
                if row["model_name"] == name
            ), None)
            if match:
                rank, row = match
                date = row.get("leaderboard_publish_date") or "读取于 " + row["read_date"]
                evidence.update(value=round(row["rating"]), public_rank=rank,
                                note=(f'{date} 开放权重组，'
                                      f'{row["vote_count"]:,} 次比较，评分区间 '
                                      f'{round(row["rating_lower"])}–{round(row["rating_upper"])}。'))
            else:
                evidence.pop("public_rank", None)
                evidence["note"] = "当前公开表未找到同名版本。历史记录：" + evidence["note"]
        by_task.setdefault(candidate["task"], []).append(candidate)

    reviews = []
    for task, group in by_task.items():
        def rank(candidate):
            by_source: dict[str, int] = {}
            independent_wins = set()
            for result in candidate["evidence"]:
                if result.get("public_rank"):
                    url = result["url"]
                    by_source[url] = min(by_source.get(url, math.inf), result["public_rank"])
                    if result["public_rank"] == 1 and (result.get("independent")
                            or result.get("arena_config") or result.get("aa_name")):
                        independent_wins.add(url)
            ranks = list(by_source.values())
            return (-len(independent_wins), -ranks.count(1), sum(ranks) / len(ranks) if ranks else math.inf,
                    -len(candidate["evidence"]))

        winner = min(group, key=rank)
        live = [e for e in winner["evidence"] if e.get("public_rank")
                and (e.get("arena_config") in snapshot or e["url"] in snapshot)]
        if live:
            winner["brief"]["effect"] = "；".join(
                f'{e["title"]} 开放权重组第 {e["public_rank"]}，{e["metric"]} {e["value"]}'
                for e in live) + "。"
        winner["alternatives"] = [{"variant": c["variant"], "evidence": c["evidence"]}
                                  for c in group if c is not winner]
        winner["review_method"] = (
            "站内 Agent 按任务比较已核对的发布版本。优先考虑独立评测中的首位次数，再比较全部来源的首位次数、平均名次与证据覆盖。"
            "各榜单的原始分数分别保留。"
            if len(group) > 1 else
            "站内 Agent 采用这组可追溯的公开评测作为参考。名次或成绩仅适用于所列测试。")
        reviews.append(winner)
    return reviews


RESEARCH_REPOS = {
    "chat": "Qwen/Qwen3-8B",
    "code": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "math": "Qwen/Qwen2.5-Math-7B-Instruct",
    "translation": "Helsinki-NLP/opus-mt-zh-en",
    "vision": "Qwen/Qwen3-VL-8B-Instruct",
    "ocr": "PaddlePaddle/PaddleOCR-VL-1.5",
    "image_gen": "stabilityai/stable-diffusion-xl-base-1.0",
    "image_edit": "Qwen/Qwen-Image-Edit",
    "speech": "openai/whisper-large-v3",
    "tts": "hexgrad/Kokoro-82M",
    "video_gen": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "video": "Qwen/Qwen2.5-VL-3B-Instruct",
    "retrieval": "BAAI/bge-m3",
}


def discover(service) -> dict:
    """Fetch metadata only. This path never schedules jobs or fetches model files."""
    api = service.source.api()

    def lookup(item):
        task, repo = item
        try:
            info = api.model_info(repo, files_metadata=True, timeout=20)
            if not info.sha:
                raise ValueError("上游未提供固定版本")
            models = service.source._to_candidates(info)
            for model in models:
                model.task = task
            return task, repo, info, models, ""
        except Exception as exc:
            return task, repo, None, [], f"{type(exc).__name__}: {str(exc)[:160]}"

    rows = []
    with ThreadPoolExecutor(max_workers=HF_MAX_WORKERS) as pool:
        for task, repo, info, models, error in pool.map(lookup, RESEARCH_REPOS.items()):
            if error:
                rows.append({"task": task, "repo": repo, "status": "来源暂不可达", "error": error})
                continue
            for model in models:
                service.db.upsert_candidate(model)
            items = [{
                "id": m.id, "repo": m.repo, "commit_rev": m.commit_rev,
                "variant": m.variant, "task": m.task, "files": [f.repo_path for f in m.files],
                "backend": m.estimate.backend, "estimate_basis": m.estimate.basis,
                "est_ram_mib": m.estimate.est_ram_mib, "evidence_count": len(m.evidence),
                "eligible": True,
            } for m in models]
            selected = LocalRuleAgent().recommend(AgentContext(candidates=items, max_suggestions=1))
            rows.append({
                "task": task, "repo": repo, "revision": info.sha, "status": "已读取元数据",
                "url": f"https://huggingface.co/{repo}/tree/{info.sha}",
                "variant": selected[0].variant if selected else None,
                "note": "公开来源候选，尚未证明 SOTA 或本机可运行" if selected else "当前格式未提取为可下载清单",
            })
    for candidate in service.db.list_candidates():
        if candidate.source == "mediapipe":
            rows.append({"task": candidate.task, "repo": candidate.repo, "revision": candidate.commit_rev,
                         "status": "官方模型已收录", "url": candidate.evidence[0].url,
                         "variant": candidate.variant, "note": "本机状态以模型库为准"})
    rows.append({"task": "face_swap", "repo": "facefusion/facefusion-assets",
                 "status": "需核对许可与来源",
                 "url": "https://github.com/facefusion/facefusion-assets",
                 "note": "换脸模型候选尚未确定可归档的权重与许可"})
    report = {"created_at": utc_now(), "method": "公开模型元数据与本地规则",
              "downloaded": False, "items": rows}
    raw_history = service.db.get_meta("agent:discovery_history")
    try:
        history = json.loads(raw_history) if raw_history else []
        if not isinstance(history, list):
            history = []
    except ValueError:
        history = []
    service.db.set_meta("agent:discovery_history", json.dumps((history + [report])[-10:], ensure_ascii=False))
    service.db.set_meta("agent:discovery", json.dumps(report, ensure_ascii=False))
    return report


def latest(service) -> dict | None:
    raw = service.db.get_meta("agent:discovery")
    try:
        return json.loads(raw) if raw else None
    except ValueError:
        return None
