"""Intent discovery with HDBSCAN. Unlike k-means, HDBSCAN finds the natural number of clusters
and marks unclustered points as noise (label -1) — that noise bucket is the long tail where the
juiciest edge cases live, so we keep it as its own 'intent'.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.cluster import HDBSCAN

from .embed import embed, tokens


def cluster_intents(questions: list[str], min_cluster_size: int = 5) -> tuple[np.ndarray, dict[int, str]]:
    """Returns (labels, intent_names). Label -1 = long-tail/outlier bucket."""
    vecs = embed(questions)
    labels = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean",
                     copy=False).fit_predict(vecs)

    # name each cluster by its most distinctive frequent tokens
    names: dict[int, str] = {-1: "long-tail"}
    for label in set(labels):
        if label == -1:
            continue
        idx = np.where(labels == label)[0]
        counter: Counter[str] = Counter()
        for i in idx:
            counter.update(set(tokens(questions[i])))
        top = [w for w, _ in counter.most_common(3)]
        names[int(label)] = "/".join(top) if top else f"intent-{label}"
    return labels, names


def cluster_sizes(labels: np.ndarray) -> dict[int, int]:
    return {int(k): int(v) for k, v in Counter(labels.tolist()).items()}
