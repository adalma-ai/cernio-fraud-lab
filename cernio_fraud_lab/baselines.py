"""Small pure-Python baselines.

These baselines avoid native dependencies so the benchmark runs on restricted
laptops and CI workers. They are intentionally simple and deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .synthetic import FEATURES


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1 / (1 + z)
    z = math.exp(value)
    return z / (1 + z)


def rule_score(x: list[list[float]]) -> list[float]:
    """Transparent analyst rule score scaled to [0, 1]."""

    scores = []
    index = {name: i for i, name in enumerate(FEATURES)}
    for row in x:
        points = 0.0
        points += min(3.0, max(0.0, math.log(row[index["amount_to_p90_ratio"]] + 1e-6) + 0.5))
        points += 1.5 * row[index["first_time_payee"]]
        points += 1.2 if row[index["payee_age_days"]] < 14 else 0.0
        points += 1.2 if row[index["rail_code"]] in {2.0, 4.0, 5.0} else 0.0
        points += 1.4 * row[index["new_device"]]
        points += 1.0 if row[index["days_since_contact_change"]] < 7 else 0.0
        points += 0.8 if row[index["outbound_count_24h"]] >= 3 else 0.0
        points += 0.5 if row[index["hour_local"]] < 6 or row[index["hour_local"]] >= 22 else 0.0
        points += 0.02 * row[index["balance_drawdown_pct"]]
        scores.append(_sigmoid(points - 5.0))
    return scores


def transform(row: list[float]) -> list[float]:
    """Numeric transform plus one-hot rail encoding for linear models."""

    i = {name: n for n, name in enumerate(FEATURES)}
    hour = row[i["hour_local"]] / 24 * 2 * math.pi
    rails = [1.0 if row[i["rail_code"]] == code else 0.0 for code in range(6)]
    return [
        math.log1p(row[i["amount_to_p90_ratio"]]),
        math.log1p(row[i["payee_age_days"]]),
        row[i["first_time_payee"]],
        row[i["outbound_count_24h"]],
        math.sin(hour),
        math.cos(hour),
        row[i["balance_drawdown_pct"]] / 100,
        math.log1p(row[i["days_since_contact_change"]]),
        row[i["new_device"]],
        math.log1p(row[i["dormant_days_before"]]),
        *rails,
    ]


@dataclass
class LogisticRegression:
    iterations: int = 700
    learning_rate: float = 0.08
    l2: float = 0.002
    weights: list[float] = field(default_factory=list)
    means: list[float] = field(default_factory=list)
    scales: list[float] = field(default_factory=list)

    def _scaled(self, x: list[list[float]]) -> list[list[float]]:
        raw = [transform(row) for row in x]
        return [[(v - m) / s for v, m, s in zip(row, self.means, self.scales)] for row in raw]

    def fit(self, x: list[list[float]], y: list[int]) -> "LogisticRegression":
        raw = [transform(row) for row in x]
        width = len(raw[0])
        self.means = [sum(row[j] for row in raw) / len(raw) for j in range(width)]
        self.scales = [
            max(1e-9, math.sqrt(sum((row[j] - self.means[j]) ** 2 for row in raw) / len(raw)))
            for j in range(width)
        ]
        data = self._scaled(x)
        self.weights = [0.0] * (width + 1)
        n = len(data)
        for _ in range(self.iterations):
            gradient = [0.0] * len(self.weights)
            for row, label in zip(data, y):
                error = _sigmoid(self.weights[0] + sum(w * v for w, v in zip(self.weights[1:], row))) - label
                gradient[0] += error
                for j, value in enumerate(row, 1):
                    gradient[j] += error * value
            for j in range(len(self.weights)):
                penalty = self.l2 * self.weights[j] if j else 0.0
                self.weights[j] -= self.learning_rate * (gradient[j] / n + penalty)
        return self

    def predict_proba(self, x: list[list[float]]) -> list[float]:
        return [_sigmoid(self.weights[0] + sum(w * v for w, v in zip(self.weights[1:], row))) for row in self._scaled(x)]


@dataclass
class BoostedStumps:
    rounds: int = 120
    learning_rate: float = 0.12
    stumps: list[tuple[int, float, float, float]] = field(default_factory=list)
    base: float = 0.0

    def fit(self, x: list[list[float]], y: list[int]) -> "BoostedStumps":
        positive_rate = min(1 - 1e-6, max(1e-6, sum(y) / len(y)))
        self.base = math.log(positive_rate / (1 - positive_rate))
        margins = [self.base] * len(x)
        thresholds = []
        for j in range(len(x[0])):
            values = sorted({row[j] for row in x})
            if len(values) <= 12:
                thresholds.append(values)
            else:
                thresholds.append([values[int(q * (len(values) - 1) / 20)] for q in range(1, 20)])
        for _ in range(self.rounds):
            residuals = [label - _sigmoid(margin) for label, margin in zip(y, margins)]
            best: tuple[float, int, float, float, float] | None = None
            for j, candidates in enumerate(thresholds):
                for threshold in candidates:
                    left = [r for row, r in zip(x, residuals) if row[j] <= threshold]
                    right = [r for row, r in zip(x, residuals) if row[j] > threshold]
                    if not left or not right:
                        continue
                    left_value = sum(left) / len(left)
                    right_value = sum(right) / len(right)
                    loss = sum((r - left_value) ** 2 for r in left) + sum((r - right_value) ** 2 for r in right)
                    if best is None or loss < best[0]:
                        best = (loss, j, threshold, left_value, right_value)
            if best is None:
                break
            _, feature, threshold, left_value, right_value = best
            self.stumps.append((feature, threshold, left_value, right_value))
            for i, row in enumerate(x):
                margins[i] += self.learning_rate * (left_value if row[feature] <= threshold else right_value)
        return self

    def predict_proba(self, x: list[list[float]]) -> list[float]:
        scores = []
        for row in x:
            margin = self.base
            for feature, threshold, left_value, right_value in self.stumps:
                margin += self.learning_rate * (left_value if row[feature] <= threshold else right_value)
            scores.append(_sigmoid(margin))
        return scores
