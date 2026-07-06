"""Build the eval set, score difficulty, and emit the coverage heatmap + the difficulty proof.

  python -m app.cli build [--traffic 60] [--k 4] [--augment-intents 4]
"""
from __future__ import annotations

import argparse
import json
import os

from .coverage import render_html, render_text, thin_cells
from .dataset import save_dataset
from .generate import build_eval_set

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    p = argparse.ArgumentParser(prog="eval-generator")
    p.add_argument("cmd", nargs="?", default="build", choices=["build"])
    p.add_argument("--traffic", type=int, default=60)
    p.add_argument("--k", type=int, default=4, help="self-consistency samples per case")
    p.add_argument("--augment-intents", type=int, default=4)
    p.add_argument("--version", default="2026-07-06")
    args = p.parse_args()

    cases, queue, report = build_eval_set(
        traffic_n=args.traffic, k=args.k, augment_intents=args.augment_intents)

    out_json = os.path.join(HERE, "eval_set.json")
    out_html = os.path.join(HERE, "coverage.html")
    save_dataset(out_json, cases, args.version)
    render_html(cases, out_html)
    with open(os.path.join(HERE, "build_report.json"), "w", encoding="utf-8") as f:
        json.dump(report.__dict__, f, indent=1)

    print(f"mined {report.traffic_count} traffic questions → {report.final_count} labeled cases")
    print(f"intents discovered: {report.intents}  (+ long-tail bucket)")
    print(f"by source: {report.by_source}")
    print(f"by measured difficulty tier: {report.by_tier}")
    print(f"review queue (low-confidence / ambiguous): {report.review_queue_size} cases\n")

    print("── difficulty proof: does self-consistency predict target-model failure? ──")
    print(f"{'tier':<8} {'n':>4} {'failures':>9} {'failure rate':>13}")
    for tier in ("easy", "medium", "hard"):
        s = report.proof.get(tier)
        if s:
            print(f"{tier:<8} {s['n']:>4} {s['failures']:>9} {s['failure_rate']:>12.0%}")
    print()

    print("coverage heatmap (intent × measured difficulty tier):")
    print(render_text(cases))

    gaps = thin_cells(cases)
    print(f"\nthin cells (<3): {len(gaps)}")

    print(f"\nwrote {out_json}, {out_html}, build_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
