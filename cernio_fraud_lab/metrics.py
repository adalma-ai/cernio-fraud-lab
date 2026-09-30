"""Metrics for imbalanced review queues."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


def average_precision(y_true: list[int], scores: list[float]) -> float:
    positives = sum(y_true)
    if positives == 0:
        return 0.0
    ordered = sorted(zip(scores, y_true), key=lambda item: item[0], reverse=True)
    hits = 0
    total = 0.0
    for rank, (_, label) in enumerate(ordered, 1):
        if label:
            hits += 1
            total += hits / rank
    return total / positives


def roc_auc(y_true: list[int], scores: list[float]) -> float:
    pairs = sorted(zip(scores, y_true), key=lambda item: item[0])
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        average_rank = (i + 1 + j) / 2
        rank_sum += average_rank * sum(label for _, label in pairs[i:j])
        i = j
    positives = sum(y_true)
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return 0.0
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def brier(y_true: list[int], probabilities: list[float]) -> float:
    return sum((p - y) ** 2 for y, p in zip(y_true, probabilities)) / len(y_true)


def top_budget(y_true: list[int], scores: list[float], budget: float) -> dict[str, float]:
    k = max(1, math.ceil(len(y_true) * budget))
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    caught = sum(y_true[i] for i in order)
    return {"alerts": k, "caught": caught, "recall": caught / max(1, sum(y_true)), "precision": caught / k}


def alerts_for_recall(y_true: list[int], scores: list[float], target: float = 0.8) -> int:
    needed = math.ceil(sum(y_true) * target)
    caught = 0
    for rank, index in enumerate(sorted(range(len(scores)), key=lambda i: scores[i], reverse=True), 1):
        caught += y_true[index]
        if caught >= needed:
            return rank
    return len(scores)


def best_f1_threshold(y_true: list[int], scores: list[float]) -> float:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in sorted(set(scores)):
        predicted = [score >= threshold for score in scores]
        tp = sum(p and y for p, y in zip(predicted, y_true))
        fp = sum(p and not y for p, y in zip(predicted, y_true))
        fn = sum((not p) and y for p, y in zip(predicted, y_true))
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold


def threshold_metrics(y_true: list[int], scores: list[float], threshold: float) -> dict[str, float]:
    predicted = [score >= threshold for score in scores]
    tp = sum(p and y for p, y in zip(predicted, y_true))
    fp = sum(p and not y for p, y in zip(predicted, y_true))
    fn = sum((not p) and y for p, y in zip(predicted, y_true))
    alerts = tp + fp
    return {
        "threshold": threshold,
        "alerts": alerts,
        "alert_rate": alerts / len(y_true),
        "precision": tp / max(1, alerts),
        "recall": tp / max(1, tp + fn),
    }


def typology_recall(rows: list[dict[str, Any]], scores: list[float], budget: float) -> dict[str, float]:
    k = max(1, math.ceil(len(rows) * budget))
    selected = set(sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k])
    totals: dict[str, int] = defaultdict(int)
    caught: dict[str, int] = defaultdict(int)
    for i, row in enumerate(rows):
        if row["label"]:
            totals[row["typology"]] += 1
            caught[row["typology"]] += int(i in selected)
    return {name: caught[name] / totals[name] for name in sorted(totals)}


def summarize(rows: list[dict[str, Any]], scores: list[float], threshold: float) -> dict[str, Any]:
    y_true = [int(row["label"]) for row in rows]
    return {
        "average_precision": average_precision(y_true, scores),
        "roc_auc": roc_auc(y_true, scores),
        "brier": brier(y_true, scores),
        "top_5pct": top_budget(y_true, scores, 0.05),
        "top_10pct": top_budget(y_true, scores, 0.10),
        "alerts_for_80pct_recall": alerts_for_recall(y_true, scores, 0.8),
        "validation_f1_threshold": threshold_metrics(y_true, scores, threshold),
        "typology_recall_at_10pct": typology_recall(rows, scores, 0.10),
    }
