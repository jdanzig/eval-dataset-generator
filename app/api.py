"""Review-queue + coverage API. Builds the eval set once on startup."""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .coverage import build_matrix, render_text, thin_cells
from .generate import build_eval_set

app = FastAPI(title="Eval Dataset Generator", version="1.0.0")

_target = int(os.getenv("EVAL_TARGET", "1500"))
cases, queue, report = build_eval_set(target=_target)


class Resolution(BaseModel):
    decision: str            # confirmed | corrected | rejected
    corrected: str | None = None


@app.get("/health")
def health():
    return {"status": "ok", "cases": report.final_count}


@app.get("/stats")
def stats():
    return {"seed_count": report.seed_count, "final_count": report.final_count,
            "intents": report.intents, "by_source": report.by_source,
            "review_queue_size": report.review_queue_size}


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
