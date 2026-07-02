"""Build the eval set and emit the coverage heatmap.

  python -m app.cli build [--traffic 500] [--seed 50] [--target 2000]
"""
from __future__ import annotations

import argparse
import os

from .coverage import render_html, render_text, thin_cells
from .dataset import save_dataset
from .generate import build_eval_set

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    p = argparse.ArgumentParser(prog="eval-generator")
    p.add_argument("cmd", nargs="?", default="build", choices=["build"])
    p.add_argument("--traffic", type=int, default=500)
    p.add_argument("--seed", type=int, default=50)
    p.add_argument("--target", type=int, default=2000)
    p.add_argument("--version", default="2026-07-01")
    args = p.parse_args()

    cases, queue, report = build_eval_set(args.traffic, args.seed, args.target)

    out_json = os.path.join(HERE, "eval_set.json")
    out_html = os.path.join(HERE, "coverage.html")
    save_dataset(out_json, cases, args.version)
    render_html(cases, out_html)

    print(f"grew eval set: {report.seed_count} seed → {report.final_count} labeled cases")
    print(f"intents discovered: {report.intents}  (+ long-tail bucket)")
    print(f"by source: {report.by_source}")
    print(f"review queue (low-confidence): {report.review_queue_size} cases\n")

    print("coverage heatmap (intent × difficulty):")
    print(render_text(cases))

    gaps = thin_cells(cases)
    print(f"\nthin cells still needing coverage (<3): {len(gaps)}")
    for intent, diff, n in gaps[:8]:
        print(f"   {intent} / {diff}: {n}")

    print(f"\nwrote {out_json} and {out_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
