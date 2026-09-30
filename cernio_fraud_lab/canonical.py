"""BIAN-aligned public contracts for the synthetic demo.

This module is a small public illustration, not a copy of any private model.
Fields mapped to a BIAN business object carry a ``bom_path``. Fraud features,
model output, and gateway references that BIAN does not define are explicit
extensions under ``cernio_lab``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

BIAN_FIELD_MAP = {
    "PaymentOrder.payment_order_id": "Payments.PaymentOrder.PaymentOrderInstruction.Identifier",
    "PaymentOrder.rail": "Payments.PaymentOrder.PaymentOrderInstruction.PaymentMechanism",
    "PaymentOrder.direction": "Payments.PaymentOrder.PaymentOrderInstruction.PaymentDirection",
    "PaymentOrder.status": "Payments.PaymentOrder.PaymentOrderInstruction.PaymentStatus",
    "PaymentOrder.amount": "Payments.PaymentOrder.PaymentOrderInstruction.PaymentAmount",
    "PaymentOrder.originator": "Payments.PaymentOrder.PaymentOrderInstruction.PaymentOriginator",
    "PaymentOrder.beneficiary": "Payments.PaymentOrder.PaymentOrderInstruction.PaymentBeneficiary",
    "PaymentOrder.remittance_info": "Payments.PaymentOrder.PaymentOrderInstruction.RemittanceInformation",
    "PaymentParty.name": "Payments.PaymentOrder.PaymentOrderInstruction.PaymentParty.Name",
    "PaymentParty.account_id": "Payments.PaymentOrder.PaymentOrderInstruction.PaymentParty.AccountIdentifier",
    "Party.party_id": "Party.PartyReferenceDataDirectory.PartyReference.Identifier",
    "CustomerCase.case_id": "CustomerCase.CustomerCaseManagement.Case.Identifier",
    "CustomerCase.status": "CustomerCase.CustomerCaseManagement.Case.CaseStatus",
    "CustomerCase.severity": "CustomerCase.CustomerCaseManagement.Case.Severity",
    "CustomerCase.subject_party_id": "CustomerCase.CustomerCaseManagement.Case.SubjectPartyReference",
    "CustomerCase.narrative": "CustomerCase.CustomerCaseManagement.Case.CaseNarrative",
}

EXTENSIONS = {
    "cernio_lab.behavior_context": "Pre-release behavioral features derived from account history.",
    "cernio_lab.beneficiary_category": "Synthetic merchant or beneficiary category used for scam typologies.",
    "cernio_lab.fraud_evaluation": "Model score, queue rank, and observable evidence. Not a BIAN BOM object.",
    "cernio_lab.core_account_ref": "Opaque core gateway account reference. Its format belongs to the private adapter.",
    "cernio_lab.hold_recommendation": "Human-approved single-payment hold proposal. Never executes in this demo.",
}

RAIL_BY_CODE = {
    0: ("PAYMENT_RAIL_ACH", "consumer_payee"),
    1: ("PAYMENT_RAIL_CARD", "merchant"),
    2: ("PAYMENT_RAIL_WIRE_DOMESTIC", "consumer_payee"),
    3: ("PAYMENT_RAIL_ZELLE", "consumer_payee"),
    4: ("PAYMENT_RAIL_WIRE_DOMESTIC", "crypto_exchange"),
    5: ("PAYMENT_RAIL_CARD", "gift_card_merchant"),
}


@dataclass(frozen=True)
class Money:
    currency_code: str
    amount_minor: int


@dataclass(frozen=True)
class PaymentParty:
    name: str
    account_id: str
    institution_id: str = "SYNTHETIC-FI"
    country_code: str = "US"
    resolved_party_id: str = ""


@dataclass(frozen=True)
class PaymentOrder:
    payment_order_id: str
    rail: str
    direction: str
    status: str
    amount: Money
    originator: PaymentParty
    beneficiary: PaymentParty
    remittance_info: str
    initiated_at: str


@dataclass(frozen=True)
class AccountReference:
    account_id: str
    party_id: str
    account_type: str
    core_account_ref: str


@dataclass(frozen=True)
class BehaviorContext:
    payment_order_id: str
    account_id: str
    p90_outbound_amount_minor: int
    payee_age_days: float
    first_time_payee: bool
    outbound_count_24h: int
    local_hour: int
    balance_drawdown_pct: float
    days_since_contact_change: float
    new_device: bool
    dormant_days_before: float
    beneficiary_category: str


@dataclass(frozen=True)
class FraudEvaluation:
    evaluation_id: str
    payment_order_id: str
    model: str
    score: float
    queue_rank: int
    evidence: list[str]


@dataclass(frozen=True)
class HoldRecommendation:
    payment_order_id: str
    account_ref: str
    scope: str
    amount: Money
    counterparty: str
    approval_required: bool = True
    executed: bool = False


@dataclass(frozen=True)
class CustomerCase:
    case_id: str
    case_type: str
    status: str
    severity: str
    subject_party_id: str
    narrative: str


def to_dict(value: Any) -> dict[str, Any]:
    return asdict(value)


def features_from_canonical(order: dict[str, Any], context: dict[str, Any]) -> dict[str, float]:
    """Build model inputs only from the canonical order and pre-release context."""

    rail_code = next(
        code
        for code, (rail, category) in RAIL_BY_CODE.items()
        if rail == order["rail"] and category == context["beneficiary_category"]
    )
    return {
        "amount_to_p90_ratio": round(order["amount"]["amount_minor"] / context["p90_outbound_amount_minor"], 4),
        "payee_age_days": float(context["payee_age_days"]),
        "first_time_payee": float(context["first_time_payee"]),
        "outbound_count_24h": float(context["outbound_count_24h"]),
        "hour_local": float(context["local_hour"]),
        "rail_code": float(rail_code),
        "balance_drawdown_pct": float(context["balance_drawdown_pct"]),
        "days_since_contact_change": float(context["days_since_contact_change"]),
        "new_device": float(context["new_device"]),
        "dormant_days_before": float(context["dormant_days_before"]),
    }
