"""Ingest a public dataset (SQuAD) as stand-in production traffic: real questions, with
reference answers and a topic (the article title) we can use as a weak label. Cached to disk.
"""
from __future__ import annotations

import json
import os
import urllib.request

_URL = ("https://datasets-server.huggingface.co/rows"
        "?dataset=rajpurkar%2Fsquad&config=plain_text&split=validation&offset={off}&length={n}")
_CACHE = os.path.join(os.path.dirname(__file__), "..", ".traffic_cache.json")


def load_traffic(n: int = 500) -> list[dict]:
    cache = os.path.abspath(_CACHE)
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            data = json.load(f)
        if len(data) >= n:
            return data[:n]

    # SQuAD rows are grouped by article, so contiguous rows share one topic. Pull from spread-out
    # offsets to get diverse topics (the dataset has ~10k validation rows across ~35 articles).
    rows: list[dict] = []
    n_windows = max(1, min(20, n // 25))
    per_window = max(25, n // n_windows)
    stride = 10570 // n_windows
    for w in range(n_windows):
        if len(rows) >= n:
            break
        base = w * stride
        fetched = 0
        while fetched < per_window and len(rows) < n:
            batch = min(100, per_window - fetched, n - len(rows))
            with urllib.request.urlopen(_URL.format(off=base + fetched, n=batch), timeout=30) as resp:
                payload = json.load(resp)
            got = payload.get("rows", [])
            if not got:
                break
            for r in got:
                row = r["row"]
                answers = row.get("answers", {}).get("text", [])
                rows.append({"question": row["question"],
                             "reference": answers[0] if answers else "",
                             "topic": row.get("title", "")})
            fetched += batch

    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(rows, f)
    return rows[:n]
