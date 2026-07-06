"""Coverage heatmap: intent × difficulty tier. Renders a terminal table and a standalone HTML
file, and flags thin cells so you can see exactly where the eval set is under-covered. Difficulty
tiers here are the MEASURED self-consistency tiers, not assigned labels.
"""
from __future__ import annotations

from collections import defaultdict

from .dataset import EvalCase

DIFFICULTIES = ["easy", "medium", "hard"]


def build_matrix(cases: list[EvalCase]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: {d: 0 for d in DIFFICULTIES})
    for c in cases:
        matrix[c.intent][c.tier] += 1
    return matrix


def render_text(cases: list[EvalCase]) -> str:
    matrix = build_matrix(cases)
    intents = sorted(matrix, key=lambda i: -sum(matrix[i].values()))
    width = max((len(i) for i in intents), default=6)
    lines = [f"{'intent':<{width}} | " + " ".join(f"{d:>7}" for d in DIFFICULTIES) + " |   total"]
    lines.append("-" * len(lines[0]))
    for intent in intents:
        row = matrix[intent]
        total = sum(row.values())
        cells = " ".join(f"{row[d]:>7}" for d in DIFFICULTIES)
        lines.append(f"{intent:<{width}} | {cells} | {total:>7}")
    return "\n".join(lines)


def thin_cells(cases: list[EvalCase], min_per_cell: int = 3) -> list[tuple[str, str, int]]:
    matrix = build_matrix(cases)
    gaps = []
    for intent, row in matrix.items():
        for d in DIFFICULTIES:
            if row[d] < min_per_cell:
                gaps.append((intent, d, row[d]))
    return gaps


def render_html(cases: list[EvalCase], path: str) -> None:
    matrix = build_matrix(cases)
    intents = sorted(matrix, key=lambda i: -sum(matrix[i].values()))
    mx = max((max(row.values()) for row in matrix.values()), default=1) or 1

    def cell(n: int) -> str:
        # green intensity by count
        shade = int(230 - 150 * min(1.0, n / mx))
        return f'<td style="background:rgb({shade},245,{shade});text-align:center">{n}</td>'

    rows = ""
    for intent in intents:
        r = matrix[intent]
        rows += (f"<tr><th style='text-align:left'>{intent}</th>"
                 + "".join(cell(r[d]) for d in DIFFICULTIES)
                 + f"<td style='text-align:center;font-weight:bold'>{sum(r.values())}</td></tr>")
    html = (f"<html><body style='font-family:system-ui'><h2>Eval coverage — "
            f"{len(cases)} cases</h2><table border=1 cellpadding=6 style='border-collapse:collapse'>"
            f"<tr><th>intent</th>{''.join(f'<th>{d}</th>' for d in DIFFICULTIES)}<th>total</th></tr>"
            f"{rows}</table></body></html>")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
