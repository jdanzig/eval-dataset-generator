"""Review-queue + coverage API.

Serves a PREBUILT eval set (`eval_set.json`, produced by `python -m app.cli build`) so startup is
instant and free — the service doesn't call the model. The review queue is reconstructed from the
cases the labeler flagged for human review.
"""
from __future__ import annotations

import json
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .coverage import build_matrix, render_text, thin_cells
from .dataset import load_dataset
from .label import ReviewQueue

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = FastAPI(title="Eval Dataset Generator", version="2.0.0")

_path = os.path.join(HERE, "eval_set.json")
if not os.path.exists(_path):
    raise RuntimeError("eval_set.json not found — run `python -m app.cli build` first")
version, cases = load_dataset(_path)

queue = ReviewQueue()
for c in cases:
    if c.needs_review:
        queue.enqueue(c.id, c.question, c.reference, c.confidence, c.ambiguous)

_report_path = os.path.join(HERE, "build_report.json")
report = json.load(open(_report_path)) if os.path.exists(_report_path) else {}


class Resolution(BaseModel):
    decision: str            # confirmed | corrected | rejected
    corrected: str | None = None


@app.get("/health")
def health():
    return {"status": "ok", "version": version, "cases": len(cases)}


@app.get("/stats")
def stats():
    return {"version": version, "count": len(cases),
            "intents": report.get("intents"), "by_source": report.get("by_source"),
            "by_tier": report.get("by_tier"),
            "review_queue_size": len(queue.pending()),
            "difficulty_proof": report.get("proof")}


@app.get("/coverage")
def coverage(fmt: str = "json"):
    if fmt == "text":
        return {"heatmap": render_text(cases)}
    return {"matrix": build_matrix(cases), "thin_cells": thin_cells(cases)}


@app.get("/review/pending")
def pending(limit: int = 20):
    items = queue.pending()
    return {"total_pending": len(items), "items": items[:limit]}


@app.post("/review/{case_id}")
def resolve(case_id: str, res: Resolution):
    try:
        return queue.resolve(case_id, res.decision, res.corrected)
    except KeyError:
        raise HTTPException(404, "case not in review queue")
