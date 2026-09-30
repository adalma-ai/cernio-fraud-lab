"""Vendor-neutral core gateway boundary.

A private production adapter can implement this protocol for a connectivity
partner. The public project ships only a synthetic fixture adapter and never
places, releases, or cancels a hold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .canonical import HoldRecommendation, Money


class CoreGateway(Protocol):
    """Read context and propose actions at the System API boundary."""

    def payment_context(self, payment_order_id: str) -> dict[str, Any]:
        ...

    def propose_single_payment_hold(self, order: dict[str, Any], account_ref: str) -> HoldRecommendation:
        ...


@dataclass
class SyntheticCoreGateway:
    """In-memory gateway used for tests and the public demo."""

    rows: list[dict[str, Any]]

    def __post_init__(self) -> None:
        self._by_payment = {row["canonical"]["payment_order"]["payment_order_id"]: row for row in self.rows}

    def payment_context(self, payment_order_id: str) -> dict[str, Any]:
        row = self._by_payment[payment_order_id]
        return {
            "payment_order": row["canonical"]["payment_order"],
            "account": row["canonical"]["account"],
            "behavior_context": row["canonical"]["behavior_context"],
        }

    def propose_single_payment_hold(self, order: dict[str, Any], account_ref: str) -> HoldRecommendation:
        if not account_ref:
            raise ValueError("A single-payment hold proposal needs an account reference.")
        amount = order["amount"]
        return HoldRecommendation(
            payment_order_id=order["payment_order_id"],
            account_ref=account_ref,
            scope="SINGLE_PAYMENT",
            amount=Money(amount["currency_code"], int(amount["amount_minor"])),
            counterparty=order["beneficiary"]["name"],
        )
