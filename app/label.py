"""LLM auto-labeling with a confidence + a human-review queue.

A strong model proposes a reference answer for each question, rates its own confidence that the
answer is correct and the question unambiguous, and flags questions too ambiguous to have a single
right answer. Low-confidence / ambiguous cases route to a review queue instead of being trusted.
`validate_labeler` measures the labeler's ambiguity calls against a human-labeled sample.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from .llm import complete

STRONG = os.getenv("EVALGEN_STRONG_MODEL", "claude-opus-4-8")
REVIEW_THRESHOLD = 0.6


@dataclass
class Label:
    answer: str
    confidence: float
    ambiguous: bool
    needs_review: bool


def _parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def auto_label(question: str) -> Label:
    text = complete(
        STRONG,
        "You are labeling an evaluation set of growth-analytics questions. For the question, "
        "return JSON with: \"answer\" (the correct, concise reference answer), \"confidence\" "
        "(0.0-1.0, how sure you are the answer is correct and the question has a single clear "
        "answer), \"ambiguous\" (true if the question is too vague or under-specified to have one "
        "correct answer). Return ONLY the JSON.\n\n"
        f"Question: {question}",
        max_tokens=300)
    d = _parse_json(text)
    ans = str(d.get("answer", "")).strip()
    conf = float(d.get("confidence", 0.5)) if isinstance(d.get("confidence"), (int, float)) else 0.5
    ambiguous = bool(d.get("ambiguous", False))
    needs = ambiguous or conf < REVIEW_THRESHOLD or not ans
    return Label(ans, round(conf, 3), ambiguous, needs)


class ReviewQueue:
    def __init__(self):
        self._items: dict[str, dict] = {}

    def enqueue(self, case_id: str, question: str, proposed: str, confidence: float,
                ambiguous: bool):
        self._items[case_id] = {"id": case_id, "question": question, "proposed": proposed,
                                "confidence": confidence, "ambiguous": ambiguous,
                                "status": "pending"}

    def pending(self) -> list[dict]:
        return [i for i in self._items.values() if i["status"] == "pending"]

    def resolve(self, case_id: str, decision: str, corrected: str | None = None) -> dict:
        item = self._items[case_id]
        item["status"] = decision
        if corrected is not None:
            item["proposed"] = corrected
        return item

    def __len__(self) -> int:
        return len(self._items)


def validate_labeler(labeled: list[dict]) -> dict:
    """Measure the labeler's ambiguity flag against human labels. Items: {question, human_ambiguous}."""
    correct = 0
    rows = []
    for item in labeled:
        lab = auto_label(item["question"])
        agree = lab.ambiguous == bool(item["human_ambiguous"])
        correct += agree
        rows.append({"question": item["question"][:50], "human": item["human_ambiguous"],
                     "labeler": lab.ambiguous, "conf": lab.confidence})
    n = len(labeled)
    return {"n": n, "agreement": round(correct / n, 3) if n else 0.0, "rows": rows}
