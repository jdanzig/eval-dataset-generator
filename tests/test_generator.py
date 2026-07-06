"""Offline unit tests — no API key, no model download. They cover the pure logic:
tiering, persistence, the review queue, coverage matrix, proof computation, and the difficulty
math's degenerate cases. The LLM-dependent paths (labeling, sampling, judging) are exercised by
`make build`, which needs a key.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.coverage import build_matrix, thin_cells
from app.dataset import EvalCase, load_dataset, save_dataset, tier_of
from app.difficulty import self_consistency_difficulty
from app.generate import _proof
from app.label import ReviewQueue, _parse_json


def _case(cid, intent, tier, correct, needs_review=False):
    return EvalCase(cid, f"q-{cid}", intent, "traffic", "ref", 0.9, False, needs_review,
                    {"easy": 0.05, "medium": 0.2, "hard": 0.5}[tier], tier, correct)


def test_tier_thresholds():
    assert tier_of(0.0) == "easy"
    assert tier_of(0.2) == "medium"
    assert tier_of(0.5) == "hard"


def test_self_consistency_degenerate():
    # fewer than 2 non-empty answers → not enough signal → 0.0 (no model call)
    assert self_consistency_difficulty([""]) == 0.0
    assert self_consistency_difficulty(["only one"]) == 0.0
    assert self_consistency_difficulty(["a", "  "]) == 0.0


def test_parse_json_extracts_object():
    assert _parse_json('junk {"answer": "x", "confidence": 0.8} tail')["confidence"] == 0.8
    assert _parse_json("no json here") == {}


def test_review_queue_lifecycle():
    q = ReviewQueue()
    q.enqueue("c1", "why?", "because", 0.4, True)
    assert len(q.pending()) == 1
    q.resolve("c1", "confirmed")
    assert q.pending() == []
    assert len(q) == 1


def test_dataset_roundtrip(tmp_path):
    cases = [_case("a", "churn", "easy", True), _case("b", "churn", "hard", False)]
    p = str(tmp_path / "d.json")
    save_dataset(p, cases, "v1")
    version, loaded = load_dataset(p)
    assert version == "v1" and len(loaded) == 2
    assert loaded[1].tier == "hard" and loaded[1].correct is False


def test_proof_bins_failure_rate_by_tier():
    cases = [_case("e1", "i", "easy", True), _case("e2", "i", "easy", True),
             _case("h1", "i", "hard", False), _case("h2", "i", "hard", True)]
    proof = _proof(cases)
    assert proof["easy"]["failure_rate"] == 0.0
    assert proof["hard"]["failure_rate"] == 0.5


def test_coverage_matrix_and_thin_cells():
    cases = [_case("a", "churn", "easy", True), _case("b", "churn", "easy", True),
             _case("c", "ltv", "hard", False)]
    m = build_matrix(cases)
    assert m["churn"]["easy"] == 2
    assert ("ltv", "easy", 0) in thin_cells(cases, min_per_cell=3)
