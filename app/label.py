"""Auto-labeling with a confidence score + a review queue.

Confidence is heuristic (offline): a case with a concrete reference answer and clear lexical
grounding scores high and is auto-accepted; sparse/empty references score low and are routed to
a human review queue instead of being trusted blindly.
"""
from __future__ import annotations

from dataclasses import dataclass

from .embed import tokens

REVIEW_THRESHOLD = 0.6


@dataclass
class Label:
    expected: str
    confidence: float
    needs_review: bool


def auto_label(question: str, reference: str) -> Label:
    ref = reference.strip()
    if not ref:
        # no gold reference to trust -> must be reviewed
        return Label("", 0.0, needs_review=True)

    # We DO have a gold reference, so start confident and dock points for the things that make an
    # auto-label untrustworthy: an over-long/essay answer, or a vague one-token answer with no
    # grounding in the question.
    confidence = 0.85
    if len(ref) > 60:
        confidence -= 0.3                       # long answers are harder to grade exactly
    if len(tokens(ref)) <= 1 and not (set(tokens(question)) & set(tokens(ref))):
        confidence -= 0.15                      # terse + ungrounded -> slightly less certain
    confidence = round(max(0.0, min(1.0, confidence)), 3)
    return Label(ref, confidence, needs_review=confidence < REVIEW_THRESHOLD)


class ReviewQueue:
    """Holds low-confidence cases for human confirm/correct/reject."""

    def __init__(self):
        self._items: dict[str, dict] = {}

    def enqueue(self, case_id: str, question: str, proposed: str, confidence: float) -> None:
        self._items[case_id] = {"id": case_id, "question": question,
                                "proposed": proposed, "confidence": confidence,
                                "status": "pending"}

    def pending(self) -> list[dict]:
        return [i for i in self._items.values() if i["status"] == "pending"]

    def resolve(self, case_id: str, decision: str, corrected: str | None = None) -> dict:
        item = self._items[case_id]
        item["status"] = decision                 # confirmed | corrected | rejected
        if corrected is not None:
            item["proposed"] = corrected
        return item

    def __len__(self) -> int:
        return len(self._items)
