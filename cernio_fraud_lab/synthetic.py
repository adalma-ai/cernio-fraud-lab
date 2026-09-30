"""Deterministic synthetic payment data.

The generator creates fictional candidate payments. Every feature describes
information available before the payment is released. Labels and scam type are
kept outside the model feature matrix.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

from .canonical import (
    AccountReference,
    BehaviorContext,
    Money,
    PaymentOrder,
    PaymentParty,
    RAIL_BY_CODE,
    features_from_canonical,
    to_dict,
)

FEATURES = [
    "amount_to_p90_ratio",
    "payee_age_days",
    "first_time_payee",
    "outbound_count_24h",
    "hour_local",
    "rail_code",
    "balance_drawdown_pct",
    "days_since_contact_change",
    "new_device",
    "dormant_days_before",
]

RAILS = {
    0: "ACH",
    1: "Card",
    2: "Wire",
    3: "P2P",
    4: "Crypto exchange",
    5: "Gift card merchant",
}

SCAM_TYPES = [
    "romance_investment",
    "gift_card",
    "tech_support",
    "account_takeover",
    "benefits_reroute",
]

HARD_NEGATIVES = [
    "home_repair",
    "family_support",
    "travel_booking",
    "new_payroll_payee",
]


@dataclass(frozen=True)
class SplitConfig:
    """Size and prevalence for one generated split."""

    rows: int
    prevalence: float
    later_period: bool = False


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _lognormal(rng: random.Random, mu: float, sigma: float) -> float:
    return math.exp(rng.gauss(mu, sigma))


def _legitimate(rng: random.Random, later_period: bool) -> tuple[dict[str, float], str]:
    kind = "routine"
    if rng.random() < 0.18:
        kind = rng.choice(HARD_NEGATIVES)

    row = {
        "amount_to_p90_ratio": _lognormal(rng, -0.35, 0.55),
        "payee_age_days": _clip(_lognormal(rng, 6.0, 0.8), 1, 3650),
        "first_time_payee": 1.0 if rng.random() < 0.08 else 0.0,
        "outbound_count_24h": float(rng.choices([0, 1, 2, 3], [0.62, 0.27, 0.08, 0.03])[0]),
        "hour_local": float(rng.choices(range(24), [1, 1, 1, 1, 1, 2, 4, 7, 9, 10, 10, 10, 10, 9, 9, 8, 8, 7, 6, 5, 4, 3, 2, 1])[0]),
        "rail_code": float(rng.choices([0, 1, 2, 3, 4, 5], [0.31, 0.41, 0.05, 0.21, 0.005, 0.015])[0]),
        "balance_drawdown_pct": _clip(rng.betavariate(1.6, 12.0) * 100, 0, 95),
        "days_since_contact_change": _clip(_lognormal(rng, 6.2, 0.9), 1, 3650),
        "new_device": 1.0 if rng.random() < 0.035 else 0.0,
        "dormant_days_before": _clip(rng.expovariate(1 / 8), 0, 300),
    }

    if kind == "home_repair":
        row.update(
            amount_to_p90_ratio=_lognormal(rng, 1.25, 0.35),
            payee_age_days=_clip(rng.uniform(0, 12), 0, 3650),
            first_time_payee=1.0,
            rail_code=float(rng.choice([0, 2, 3])),
            balance_drawdown_pct=_clip(rng.gauss(38, 14), 5, 90),
        )
    elif kind == "family_support":
        row.update(
            amount_to_p90_ratio=_lognormal(rng, 0.8, 0.45),
            rail_code=3.0,
            outbound_count_24h=float(rng.choice([1, 2, 3])),
            balance_drawdown_pct=_clip(rng.gauss(22, 10), 3, 70),
        )
    elif kind == "travel_booking":
        row.update(
            amount_to_p90_ratio=_lognormal(rng, 0.9, 0.35),
            hour_local=float(rng.choice([21, 22, 23, 0, 1])),
            rail_code=1.0,
            new_device=1.0 if rng.random() < 0.18 else 0.0,
        )
    elif kind == "new_payroll_payee":
        row.update(
            amount_to_p90_ratio=_lognormal(rng, 0.45, 0.35),
            payee_age_days=_clip(rng.uniform(1, 30), 1, 3650),
            first_time_payee=1.0,
            rail_code=0.0,
        )

    if later_period:
        row["outbound_count_24h"] += 1.0 if rng.random() < 0.08 else 0.0
        if rng.random() < 0.04:
            row["rail_code"] = 3.0
    return row, kind


def _scam(rng: random.Random, later_period: bool) -> tuple[dict[str, float], str]:
    kind = rng.choices(SCAM_TYPES, [0.27, 0.18, 0.2, 0.2, 0.15])[0]
    row, _ = _legitimate(rng, later_period)

    if kind == "romance_investment":
        row.update(
            amount_to_p90_ratio=_lognormal(rng, 1.4, 0.5),
            payee_age_days=_clip(rng.uniform(0, 45), 0, 3650),
            first_time_payee=1.0 if rng.random() < 0.72 else 0.0,
            rail_code=float(rng.choices([2, 4, 3], [0.35, 0.5, 0.15])[0]),
            balance_drawdown_pct=_clip(rng.gauss(45, 18), 5, 99),
        )
    elif kind == "gift_card":
        row.update(
            amount_to_p90_ratio=_lognormal(rng, 0.72, 0.35),
            first_time_payee=1.0 if rng.random() < 0.55 else 0.0,
            outbound_count_24h=float(rng.choice([2, 3, 4, 5])),
            rail_code=float(rng.choices([5, 1], [0.8, 0.2])[0]),
            balance_drawdown_pct=_clip(rng.gauss(18, 9), 2, 70),
        )
    elif kind == "tech_support":
        row.update(
            amount_to_p90_ratio=_lognormal(rng, 1.05, 0.45),
            payee_age_days=_clip(rng.uniform(0, 5), 0, 3650),
            first_time_payee=1.0,
            rail_code=float(rng.choice([2, 3, 4])),
            new_device=1.0 if rng.random() < 0.45 else 0.0,
        )
    elif kind == "account_takeover":
        row.update(
            amount_to_p90_ratio=_lognormal(rng, 1.1, 0.55),
            payee_age_days=_clip(rng.uniform(0, 10), 0, 3650),
            first_time_payee=1.0 if rng.random() < 0.8 else 0.0,
            days_since_contact_change=_clip(rng.uniform(0, 5), 0, 3650),
            new_device=1.0 if rng.random() < 0.82 else 0.0,
            hour_local=float(rng.choice([0, 1, 2, 3, 4, 5, 22, 23])),
        )
    else:
        row.update(
            amount_to_p90_ratio=_lognormal(rng, 0.25, 0.35),
            payee_age_days=_clip(rng.uniform(0, 20), 0, 3650),
            first_time_payee=1.0 if rng.random() < 0.4 else 0.0,
            rail_code=0.0,
            days_since_contact_change=_clip(rng.uniform(0, 30), 0, 3650),
            dormant_days_before=_clip(rng.uniform(15, 120), 0, 300),
        )

    if later_period and kind in {"gift_card", "tech_support"} and rng.random() < 0.35:
        row["rail_code"] = 3.0
    return row, kind


def generate_split(name: str, config: SplitConfig, seed: int) -> list[dict[str, Any]]:
    """Generate one deterministic split."""

    rng = random.Random(seed)
    frauds = round(config.rows * config.prevalence)
    labels = [1] * frauds + [0] * (config.rows - frauds)
    rng.shuffle(labels)
    rows: list[dict[str, Any]] = []
    for i, label in enumerate(labels):
        features, typology = _scam(rng, config.later_period) if label else _legitimate(rng, config.later_period)
        payment_id = f"{name[:2].upper()}-{seed}-{i:05d}"
        canonical = _canonical_records(payment_id, features, rng)
        rounded = features_from_canonical(canonical["payment_order"], canonical["behavior_context"])
        rows.append(
            {
                "payment_id": payment_id,
                "split": name,
                "label": label,
                "typology": typology if label else "legitimate",
                "legit_pattern": typology if not label else "",
                "features": rounded,
                "canonical": canonical,
            }
        )
    return rows


def _canonical_records(payment_id: str, features: dict[str, float], rng: random.Random) -> dict[str, Any]:
    account_number = rng.randrange(10_000, 99_999)
    party_id = f"PTY-{account_number}"
    account_id = f"ACC-{account_number}"
    p90_minor = rng.randrange(20_000, 250_000)
    amount_minor = max(100, round(p90_minor * features["amount_to_p90_ratio"]))
    rail_code = int(features["rail_code"])
    rail, category = RAIL_BY_CODE[rail_code]
    payee_id = rng.randrange(1_000, 9_999)
    order = PaymentOrder(
        payment_order_id=payment_id,
        rail=rail,
        direction="PAYMENT_DIRECTION_OUTBOUND",
        status="PAYMENT_STATUS_PENDING",
        amount=Money("USD", amount_minor),
        originator=PaymentParty("Synthetic Member", account_id, resolved_party_id=party_id),
        beneficiary=PaymentParty(f"Synthetic Payee {payee_id}", f"PAYEE-{payee_id}"),
        remittance_info="synthetic payment",
        initiated_at=f"2026-10-01T{int(features['hour_local']):02d}:00:00-07:00",
    )
    account = AccountReference(
        account_id=account_id,
        party_id=party_id,
        account_type="ACCOUNT_TYPE_CHECKING",
        core_account_ref=f"SYNTH-CI-{account_number % 97:02d}/SYNTH-AR-{account_number}",
    )
    context = BehaviorContext(
        payment_order_id=payment_id,
        account_id=account_id,
        p90_outbound_amount_minor=p90_minor,
        payee_age_days=round(features["payee_age_days"], 4),
        first_time_payee=bool(features["first_time_payee"]),
        outbound_count_24h=int(features["outbound_count_24h"]),
        local_hour=int(features["hour_local"]),
        balance_drawdown_pct=round(features["balance_drawdown_pct"], 4),
        days_since_contact_change=round(features["days_since_contact_change"], 4),
        new_device=bool(features["new_device"]),
        dormant_days_before=round(features["dormant_days_before"], 4),
        beneficiary_category=category,
    )
    return {"payment_order": to_dict(order), "account": to_dict(account), "behavior_context": to_dict(context)}


def generate_dataset(seed: int = 7) -> dict[str, list[dict[str, Any]]]:
    """Return independently generated train, validation, and later-period test sets."""

    return {
        "train": generate_split("train", SplitConfig(1200, 0.08), seed * 101 + 1),
        "validation": generate_split("validation", SplitConfig(500, 0.08), seed * 101 + 2),
        "test": generate_split("test", SplitConfig(1000, 0.06, later_period=True), seed * 101 + 3),
    }


def matrix(rows: list[dict[str, Any]]) -> list[list[float]]:
    """Return model inputs in the fixed feature order."""

    return [[float(row["features"][name]) for name in FEATURES] for row in rows]


def labels(rows: list[dict[str, Any]]) -> list[int]:
    return [int(row["label"]) for row in rows]
