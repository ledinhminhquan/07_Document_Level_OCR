"""Command-line interface — the single entrypoint for the Document-Level OCR system.

    dococr <command> [options]

Commands: data, synth, train, tune, evaluate, ocr, correct, demo-agent, serve,
benchmark, error-analysis, robustness, monitor, generate-report, generate-slides,
autopilot, grade.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .config import AppConfig, ensure_dirs, load_config
from .logging_utils import get_logger

logger = get_logger(__name__)

TITLE = "Document-Level OCR System"
AUTHOR = "Le Dinh Minh Quan"


def _load(args) -> AppConfig:
    cfg = load_config(args.config) if getattr(args, "config", None) else AppConfig()
    ensure_dirs()
    return cfg


def cmd_data(args):
    from .data.download_dataset import download_all, download_task
    cfg = _load(args)
    res = download_all(cfg) if args.task == "all" else download_task(args.task, cfg)
    print(json.dumps(res, indent=2, ensure_ascii=False))


def cmd_synth(args):
    from .data.ocr_noise import OCRNoiseGenerator
    ex = OCRNoiseGenerator(_load(args).data).generate(args.n, seed=0)
    print(json.dumps([{"noisy": e["noisy"], "clean": e["clean"]} for e in ex], indent=2, ensure_ascii=False))


def cmd_train(args):
    from .training.train_corrector import train_corrector
    print(json.dumps(train_corrector(_load(args), limit=args.limit, base_model=args.base_model), indent=2))


def cmd_tune(args):
    from .training.tune import tune_corrector
    print(json.dumps(tune_corrector(_load(args), n_trials=args.n_trials, limit=args.limit), indent=2))


def cmd_evaluate(args):
    from .training.evaluate import evaluate
    print(json.dumps(evaluate(_load(args), which=args.which, limit=args.limit).get("summary", {}),
                     indent=2, ensure_ascii=False))


def cmd_ocr(args):
    from .agent.doc_agent import DocumentAgent
    job = DocumentAgent(_load(args), load_model=not args.stub, ocr_engine=args.engine).process(
        path=args.file, save=True)
    print(json.dumps(job.to_dict(), indent=2, ensure_ascii=False))


def cmd_correct(args):
    from .agent.doc_agent import DocumentAgent
    text = args.text or Path(args.file).read_text(encoding="utf-8")
    job = DocumentAgent(_load(args), load_model=not args.identity).process(text=text, save=False)
    print(json.dumps({"corrected": job.full_text, "markdown": job.markdown,
                      "corrections": job.metrics.get("corrections")}, indent=2, ensure_ascii=False))


def cmd_demo_agent(args):
    from .agent.doc_agent import DocumentAgent
    from .data.samples import SAMPLE_DOC
    from .data.ocr_noise import corrupt
    import random
    cfg = _load(args)
    noisy = corrupt(SAMPLE_DOC, random.Random(0), 0.08)
    job = DocumentAgent(cfg, load_model=not args.identity).process(text=noisy, filename="demo", save=False)
    sd = job.to_dict()
    print(f"\nstatus     : {sd['status']}")
    print(f"pages      : {sd['n_pages']} | blocks: {sd['n_blocks']} | flagged: {sd['n_flagged']}")
    print(f"corrector  : {sd['model_versions'].get('corrector')}")
    print(f"decisions  : {[(d['id'], d['branch']) for d in sd['decisions']]}")
    print(f"corrections: {sd['metrics'].get('corrections')}")
    print("\n--- input (simulated OCR, first 200) ---\n" + noisy[:200])
    print("\n--- output markdown (first 280) ---\n" + sd["markdown"][:280])


def cmd_serve(args):
    import os
    import uvicorn
    if args.config:
        os.environ["DOCOCR_INFER_CONFIG"] = str(args.config)
    if args.engine:
        os.environ["DOCOCR_OCR_ENGINE"] = args.engine
    target = "dococr.api.app_combined:app" if args.ui else "dococr.api.main:app"
    uvicorn.run(target, host=args.host, port=args.port, reload=False)


def cmd_benchmark(args):
    from .analysis.latency import benchmark
    print(json.dumps(benchmark(_load(args), n=args.n, warmup=args.warmup), indent=2))


def cmd_error_analysis(args):
    from .analysis.error_analysis import error_analysis
    print(json.dumps(error_analysis(_load(args), limit=args.limit), indent=2, ensure_ascii=False))


def cmd_robustness(args):
    from .analysis.robustness import robustness_report
    print(json.dumps(robustness_report(_load(args), limit=args.limit), indent=2))


def cmd_monitor(args):
    from .monitoring.drift_report import monitoring_report
    print(json.dumps(monitoring_report(_load(args), log_path=args.log), indent=2))


def cmd_generate_report(args):
    from .autoreport.report_pdf import generate_report
    print("Report ->", generate_report(_load(args), title=args.title, author=args.author))


def cmd_generate_slides(args):
    from .autoreport.slides_pptx import generate_slides
    print("Slides ->", generate_slides(_load(args), title=args.title, author=args.author))


def cmd_autopilot(args):
    from .automation.autopilot import run_autopilot
    print(json.dumps(run_autopilot(_load(args), title=args.title, author=args.author,
                                   train=not args.no_train, limit=args.limit), indent=2))


def cmd_grade(args):
    from .grading.checklist import build_checklist
    repo = Path(args.repo) if args.repo else Path(__file__).resolve().parents[2]
    print(json.dumps(build_checklist(repo), indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dococr", description=TITLE)
    p.add_argument("--config", help="Path to a YAML config")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("data", help="build the post-OCR corpus + samples"); sp.add_argument("--task", choices=["all", "corpus", "real"], default="corpus"); sp.set_defaults(func=cmd_data)
    sp = sub.add_parser("synth", help="show a few synthetic (noisy, clean) pairs"); sp.add_argument("--n", type=int, default=6); sp.set_defaults(func=cmd_synth)
    sp = sub.add_parser("train", help="fine-tune the ByT5 post-OCR corrector"); sp.add_argument("--limit", type=int, default=None); sp.add_argument("--base-model", default=None); sp.set_defaults(func=cmd_train)
    sp = sub.add_parser("tune", help="basic LR hyperparameter search"); sp.add_argument("--n-trials", type=int, default=3); sp.add_argument("--limit", type=int, default=4000); sp.set_defaults(func=cmd_tune)
    sp = sub.add_parser("evaluate", help="corrector vs baselines (CER/WER reduction)"); sp.add_argument("--which", choices=["test", "val", "real"], default="test"); sp.add_argument("--limit", type=int, default=None); sp.set_defaults(func=cmd_evaluate)
    sp = sub.add_parser("ocr", help="OCR + structure a document image/PDF"); sp.add_argument("--file", required=True); sp.add_argument("--engine", default=None); sp.add_argument("--stub", action="store_true"); sp.set_defaults(func=cmd_ocr)
    sp = sub.add_parser("correct", help="post-OCR correct raw text"); sp.add_argument("--text"); sp.add_argument("--file"); sp.add_argument("--identity", action="store_true"); sp.set_defaults(func=cmd_correct)
    sp = sub.add_parser("demo-agent", help="run the agent on a simulated-OCR sample doc"); sp.add_argument("--identity", action="store_true"); sp.set_defaults(func=cmd_demo_agent)
    sp = sub.add_parser("serve", help="start the FastAPI server"); sp.add_argument("--host", default="0.0.0.0"); sp.add_argument("--port", type=int, default=8000); sp.add_argument("--ui", action="store_true"); sp.add_argument("--engine", default=None); sp.set_defaults(func=cmd_serve)
    sp = sub.add_parser("benchmark", help="latency benchmark"); sp.add_argument("--n", type=int, default=30); sp.add_argument("--warmup", type=int, default=3); sp.set_defaults(func=cmd_benchmark)
    sp = sub.add_parser("error-analysis", help="error analysis (improved/degraded/by error-type)"); sp.add_argument("--limit", type=int, default=None); sp.set_defaults(func=cmd_error_analysis)
    sp = sub.add_parser("robustness", help="robustness to increasing OCR-noise levels"); sp.add_argument("--limit", type=int, default=80); sp.set_defaults(func=cmd_robustness)
    sp = sub.add_parser("monitor", help="monitoring report from job logs"); sp.add_argument("--log", default=None); sp.set_defaults(func=cmd_monitor)
    sp = sub.add_parser("generate-report", help="generate the PDF report"); sp.add_argument("--title", default=TITLE); sp.add_argument("--author", default=AUTHOR); sp.set_defaults(func=cmd_generate_report)
    sp = sub.add_parser("generate-slides", help="generate the PPTX slides"); sp.add_argument("--title", default=TITLE); sp.add_argument("--author", default=AUTHOR); sp.set_defaults(func=cmd_generate_slides)
    sp = sub.add_parser("autopilot", help="one-button: train -> eval -> analysis -> report+slides"); sp.add_argument("--title", default=TITLE); sp.add_argument("--author", default=AUTHOR); sp.add_argument("--no-train", action="store_true"); sp.add_argument("--limit", type=int, default=None); sp.set_defaults(func=cmd_autopilot)
    sp = sub.add_parser("grade", help="rubric completeness self-check"); sp.add_argument("--repo", default=None); sp.set_defaults(func=cmd_grade)
    return p


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
