"""同步 CLI：与网页使用同一套业务代码，网页关闭后同步仍可运行（文档 §2、§6）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .app import Service
from .config import APP_VERSION, DEFAULT_PORT, LOOPBACK_HOST
from .models import Settings


def _out(data: Any) -> None:
    if isinstance(data, (dict, list)):
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print(data)


def _service(args: argparse.Namespace) -> Service:
    return Service(data_dir=getattr(args, "data", None))


def _parse_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except ValueError:
        return raw


def cmd_serve(args, svc: Service) -> int:
    import uvicorn
    from .app import create_app
    app = create_app(svc)
    print(f"末日种子本地服务：http://{args.host}:{args.port}  （仅回环）")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level,
                access_log=False)
    return 0


def cmd_sync(args, svc: Service) -> int:
    service = svc.pipeline.sync(trigger="scheduled" if args.scheduled else "manual",
                                wait=args.wait)
    _out(service.model_dump())
    return 0 if service.status in ("ok", "partial", "skipped") else 1


def cmd_download(args, svc: Service) -> int:
    job = svc.pipeline.submit_download(args.candidate_id, args.task, args.profile,
                                       autostart=not args.wait)
    print(f"作业 {job.id} 已创建（幂等键 {job.idempotency_key}）")
    if args.wait:
        if svc.db.get_job(job.id).state == "queued":
            svc.pipeline.run_job(job.id)
        _out(svc.db.get_job(job.id).model_dump())
    else:
        print("用 jobs 命令或网页轮询进度。")
    return 0


def cmd_list(args, svc: Service) -> int:
    items = svc.db.list_candidates(task=args.task, state=args.state)
    rows = [{"id": c.id, "task": c.task, "repo": c.repo, "variant": c.variant,
             "state": c.state, "license": c.license, "size_bytes": c.size_bytes,
             "reason": c.reason} for c in items]
    _out(rows if not args.json_ndjson else [json.dumps(r, ensure_ascii=False) for r in rows])
    return 0


def cmd_hardware(args, svc: Service) -> int:
    sample = svc.hardware()
    svc.db.save_sample(sample)
    from .hardware import budget
    _out({"sample": sample.model_dump(), "budget_cpu": budget(sample, "cpu").to_dict(),
          "budget_cuda": budget(sample, "cuda").to_dict(),
          "fingerprint": sample.fingerprint()})
    return 0


def cmd_jobs(args, svc: Service) -> int:
    _out([j.model_dump() for j in svc.db.list_jobs(limit=args.limit, state=args.state)])
    return 0


def cmd_settings(args, svc: Service) -> int:
    settings = svc.db.get_settings()
    if args.set:
        data = settings.model_dump()
        for pair in args.set:
            if "=" not in pair:
                print(f"忽略无效设置：{pair}（应为 key=value）", file=sys.stderr)
                continue
            key, raw = pair.split("=", 1)
            if key not in Settings.model_fields:
                print(f"未知设置项：{key}", file=sys.stderr)
                continue
            data[key] = _parse_value(raw)
        settings = Settings(**data)
        svc.db.put_settings(settings)
    _out({**settings.model_dump(), "effective": svc.scheduler.status()})
    return 0


def cmd_default(args, svc: Service) -> int:
    db = svc.db
    pointer = db.get_default(args.task, args.profile)
    old = pointer.seed_id
    if args.auto:
        from .catalog import load_eval_config, promote, rank
        config = load_eval_config()
        pairs = []
        for record in db.list_evaluations(eval_version=config["version"]):
            seed = db.find_seed_by_candidate(record.candidate_id)
            if seed and seed.offline_ready:
                pairs.append((record, seed))
        if not pairs:
            print("没有通过门槛的本机成绩，默认指针保持不变。")
            return 1
        best = rank([r for r, _ in pairs], args.profile)[0]
        ok, reason = promote(None, best, args.profile, config)
        print(reason)
        if not ok:
            return 1
        pointer.seed_id = next(s for r, s in pairs if r.id == best.id).id
        pointer.pinned = False
        pointer.source = "auto"
        pointer.previous_seed_id = old
    else:
        seed = db.get_seed(args.seed_id or "")
        if not seed or not seed.offline_ready:
            print("只能固定或切换到「离线可用」的已验证版本。")
            return 1
        pointer.seed_id = seed.id
        pointer.pinned = not args.unpin
        pointer.previous_seed_id = old
        pointer.source = "user"
    db.set_default(pointer)
    _out(pointer.model_dump())
    return 0


def cmd_verify(args, svc: Service) -> int:
    from .archive import compute_completeness, reconcile, verify_seed
    report = reconcile(svc.db, svc.paths)
    checks = []
    for seed in svc.db.list_seeds():
        directory = svc.paths.seeds / seed.rel_dir
        ok, missing, mismatched = verify_seed(directory, seed.manifest)
        ratio, _ = compute_completeness(directory, seed.manifest)
        checks.append({"seed_id": seed.id, "ok": ok, "completeness": ratio,
                       "missing": missing, "mismatched": mismatched})
    _out({"reconcile": report, "seeds": checks,
          "all_ok": all(c["ok"] for c in checks)})
    return 0 if all(c["ok"] for c in checks) else 1


def cmd_export(args, svc: Service) -> int:
    if args.wheelhouse:
        _out(svc.recovery.build_wheelhouse())
    _out(svc.recovery.export(args.out, include_seeds=not args.no_seeds))
    return 0


def cmd_drill(args, svc: Service) -> int:
    if args.record:
        _out(svc.recovery.record_real_drill())
    elif args.simulate is not None:
        _out(svc.recovery.set_simulated(args.simulate == "on"))
    else:
        _out(svc.recovery.status())
    return 0


def cmd_schedule(args, svc: Service) -> int:
    if args.action == "install":
        _out(svc.scheduler.register(svc.db.get_settings().schedule_time))
    elif args.action == "remove":
        _out(svc.scheduler.unregister())
    elif args.verify:
        _out(svc.scheduler.verify_registration())
    else:
        _out(svc.scheduler.status())
    return 0


def cmd_runtime(args, svc: Service) -> int:
    if args.action == "register":
        if not args.command:
            print("register 需要 --command（可多次或用空格分隔的完整路径）", file=sys.stderr)
            return 2
        _out(svc.registry.register(args.backend, args.command, args.version, args.build))
    else:
        data = svc.registry.load()
        _out({"registry": data, "probe": svc.registry.probe()[0],
              "detail": svc.registry.probe()[1]})
    return 0


def cmd_retest(args, svc: Service) -> int:
    """对已归档种子按当前评测版本重测（换口径后需重测候选与现任）。"""
    from .downloader import DownloadPlan
    from .models import FileSpec, RuntimeConfig
    seed = svc.db.get_seed(args.seed_id)
    if not seed:
        print(f"种子不存在：{args.seed_id}")
        return 1
    candidate = svc.db.get_candidate(seed.candidate_id)
    directory = svc.paths.seeds / seed.rel_dir
    files = [FileSpec(repo_path=i["source_path"], local_path=i["relative_path"],
                      size_bytes=i["size_bytes"], kind=i["kind"],
                      sha256=i.get("sha256")) for i in seed.manifest.get("sizes_and_hashes", [])]
    plan = DownloadPlan(candidate_id=candidate.id, repo=candidate.repo,
                        commit_rev=seed.manifest["source"]["fixed_revision"], files=files,
                        runtime_config=RuntimeConfig(**seed.manifest["runtime_config"]))
    verification = {"files": [{"local_path": i["relative_path"], "kind": i["kind"],
                               "size_bytes": i["size_bytes"], "sha256": i.get("sha256"),
                               "repo_path": i["source_path"]}
                              for i in seed.manifest.get("sizes_and_hashes", [])],
                    "total_bytes": seed.manifest.get("total_bytes", 0)}
    record = svc.evaluator.evaluate(candidate=candidate, staging_dir=directory, plan=plan,
                                    config=plan.runtime_config, verification=verification)
    _out(record.model_dump())
    return 0


def cmd_doctor(args, svc: Service) -> int:
    from .archive import verify_seed
    from .catalog import load_eval_config
    hardware = svc.hardware()
    backends, detail = svc.registry.probe()
    seeds = svc.db.list_seeds()
    problems = []
    if not any(backends.values()):
        problems.append("没有已登记的可用运行器；作业会标「待适配」")
    if not (svc.paths.web / "index.html").exists():
        problems.append("缺少本地网页 web/index.html")
    if not (svc.paths.web_assets / "styles.css").exists():
        problems.append("缺少本地样式 web/assets/styles.css")
    if not (svc.paths.web_assets / "app.js").exists():
        problems.append("缺少本地脚本 web/assets/app.js")
    if not (svc.paths.web_assets / "vault.webp").exists():
        problems.append("缺少本地背景图 web/assets/vault.webp（须导出为 WebP）")
    for seed in seeds:
        ok, missing, _ = verify_seed(svc.paths.seeds / seed.rel_dir, seed.manifest)
        if not ok:
            problems.append(f"种子 {seed.id} 缺件：{missing[:2]}")
    _out({
        "version": APP_VERSION, "data_dir": str(svc.paths.data),
        "hardware": {"cpu": hardware.cpu_model, "arch": hardware.arch,
                     "ram_mib": round(hardware.ram_total_mib),
                     "vram_mib": round(hardware.vram_total_mib),
                     "instruction_sets": hardware.instruction_sets},
        "backends": backends, "backend_detail": detail,
        "eval_version": load_eval_config()["version"],
        "seeds": len(seeds), "candidates": len(svc.db.list_candidates()),
        "startup_reconcile": svc.startup(),
        "schedule": svc.scheduler.status(),
        "problems": problems,
        "ok": not problems,
    })
    return 0 if not problems else 1


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data", help="数据目录（默认项目 data/；离线恢复可指向别的目录）")
    parser = argparse.ArgumentParser(prog="doomsday-seed", description="末日种子 · 本地 CLI")
    parser.add_argument("--version", action="version", version=APP_VERSION)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("serve", parents=[common], help="启动本地服务与网页")
    p.add_argument("--host", default=LOOPBACK_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--log-level", default="warning")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("sync", parents=[common], help="发现、预选并下载入围候选")
    p.add_argument("--scheduled", action="store_true", help="标记为计划任务触发")
    p.add_argument("--wait", action="store_true", help="等待作业完成（计划任务不要用）")
    p.set_defaults(func=cmd_sync)

    p = sub.add_parser("download", parents=[common], help="下载指定候选")
    p.add_argument("candidate_id")
    p.add_argument("--task", default="chat")
    p.add_argument("--profile", default="balanced")
    p.add_argument("--wait", action="store_true")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("list", parents=[common], help="列出候选目录")
    p.add_argument("--task")
    p.add_argument("--state")
    p.add_argument("--json-ndjson", action="store_true")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("hardware", parents=[common], help="采样本机硬件与资源预算")
    p.set_defaults(func=cmd_hardware)

    p = sub.add_parser("jobs", parents=[common], help="查看作业队列")
    p.add_argument("--state")
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(func=cmd_jobs)

    p = sub.add_parser("settings", parents=[common], help="读写订阅与计划设置")
    p.add_argument("--set", action="append", metavar="KEY=VALUE")
    p.set_defaults(func=cmd_settings)

    p = sub.add_parser("default", parents=[common], help="固定/切换/自动推荐默认版本")
    p.add_argument("task")
    p.add_argument("profile")
    p.add_argument("seed_id", nargs="?")
    p.add_argument("--auto", action="store_true")
    p.add_argument("--unpin", action="store_true")
    p.set_defaults(func=cmd_default)

    p = sub.add_parser("verify", parents=[common], help="启动对账与 SHA-256 校验")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("retest", parents=[common], help="按当前评测版本重测已归档种子")
    p.add_argument("seed_id")
    p.set_defaults(func=cmd_retest)

    p = sub.add_parser("export", parents=[common], help="导出离线恢复包")
    p.add_argument("--out")
    p.add_argument("--no-seeds", action="store_true")
    p.add_argument("--wheelhouse", action="store_true", help="先 pip download 锁定依赖")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("drill", parents=[common], help="断网演练步骤与记录")
    p.add_argument("--simulate", choices=["on", "off"])
    p.add_argument("--record", action="store_true", help="记录已完成系统层断网演练")
    p.set_defaults(func=cmd_drill)

    p = sub.add_parser("schedule", parents=[common], help="Windows 任务计划")
    p.add_argument("action", choices=["install", "remove", "status"], nargs="?", default="status")
    p.add_argument("--verify", action="store_true", help="核对任务计划程序中的实际注册状态")
    p.set_defaults(func=cmd_schedule)

    p = sub.add_parser("runtime", parents=[common], help="锁定/查看运行器构建")
    p.add_argument("action", choices=["show", "register"], nargs="?", default="show")
    p.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--command", nargs="+")
    p.add_argument("--version", default="b0000")
    p.add_argument("--build", default="")
    p.set_defaults(func=cmd_runtime)

    p = sub.add_parser("doctor", parents=[common], help="综合自检")
    p.set_defaults(func=cmd_doctor)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    svc = _service(args)
    try:
        return int(args.func(args, svc) or 0)
    except KeyboardInterrupt:
        return 130
    finally:
        if args.command != "serve":
            svc.shutdown()
