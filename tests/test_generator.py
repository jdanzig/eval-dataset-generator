import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.augment import adversarials, paraphrases
from app.coverage import build_matrix, thin_cells
from app.generate import build_eval_set
from app.label import auto_label


def synthetic_traffic(n=120):
    topics = [
        ("physics", "What is the speed of light?", "300000 km/s"),
        ("history", "When did the French Revolution begin?", "1789"),
        ("biology", "What organelle produces energy in a cell?", "mitochondria"),
        ("geography", "What is the capital of Japan?", "Tokyo"),
    ]
    out = []
    for i in range(n):
        topic, q, a = topics[i % len(topics)]
        # vary the question slightly so clusters have >1 member but stay topical
        out.append({"question": f"{q} (item {i // len(topics)})", "reference": a, "topic": topic})
    return out


def test_confidence_high_with_reference():
    lab = auto_label("What is the capital of Japan?", "Tokyo")
    assert lab.confidence >= 0.6
    assert not lab.needs_review


def test_empty_reference_goes_to_review():
    lab = auto_label("Unanswerable?", "")
    assert lab.needs_review
    assert lab.confidence == 0.0


def test_augment_variants_are_distinct():
    q = "What is the capital of Japan?"
    ps, advs = paraphrases(q), adversarials(q)
    assert len(set(ps)) == len(ps) and all(ps)
    assert len(set(advs)) == len(advs) and all(advs)


def test_build_reaches_target_and_records_seed():
    cases, queue, report = build_eval_set(seed_n=20, target=300, min_cluster_size=5,
                                          traffic=synthetic_traffic(120))
    assert report.seed_count == 20
    assert report.final_count >= 300
    # difficulty bands all present
    diffs = {c.difficulty for c in cases}
    assert {"easy", "medium", "hard"} <= diffs


def test_coverage_balanced_no_thin_cells_after_growth():
    cases, _, _ = build_eval_set(seed_n=20, target=400, traffic=synthetic_traffic(120))
    matrix = build_matrix(cases)
    assert len(matrix) >= 2                    # multiple intents
    # round-robin augmentation should leave few/no thin cells among non-tiny intents
    assert len(thin_cells(cases, min_per_cell=3)) <= 2
