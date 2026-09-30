from __future__ import annotations

import json

import pytest

from cernio_fraud_lab.baselines import BoostedStumps, LogisticRegression, rule_score
from cernio_fraud_lab.canonical import BIAN_FIELD_MAP, features_from_canonical
from cernio_fraud_lab.gateway import SyntheticCoreGateway
from cernio_fraud_lab.metrics import average_precision, roc_auc, top_budget
from cernio_fraud_lab.synthetic import FEATURES, generate_dataset, labels, matrix
from cernio_fraud_lab.tabpfn_api import TabPFNAPIError, TabPFNRestClassifier


def test_generator_is_deterministic_and_keeps_labels_out_of_features() -> None:
    first = generate_dataset(3)
    second = generate_dataset(3)
    assert first == second
    assert list(first["train"][0]["features"]) == FEATURES
    assert "label" not in FEATURES
    assert "typology" not in FEATURES
    assert {name: sum(labels(rows)) for name, rows in first.items()} == {"train": 96, "validation": 40, "test": 60}


def test_splits_are_independent_and_include_all_scam_types() -> None:
    data = generate_dataset(5)
    ids = [row["payment_id"] for rows in data.values() for row in rows]
    assert len(ids) == len(set(ids))
    assert {row["typology"] for row in data["train"] if row["label"]} == {
        "account_takeover",
        "benefits_reroute",
        "gift_card",
        "romance_investment",
        "tech_support",
    }


def test_model_features_are_derived_from_bian_aligned_records() -> None:
    row = generate_dataset(4)["test"][0]
    canonical = row["canonical"]
    assert features_from_canonical(canonical["payment_order"], canonical["behavior_context"]) == row["features"]
    assert canonical["payment_order"]["direction"] == "PAYMENT_DIRECTION_OUTBOUND"
    assert BIAN_FIELD_MAP["PaymentOrder.amount"].startswith("Payments.PaymentOrder")


def test_gateway_only_proposes_a_single_payment_hold() -> None:
    rows = generate_dataset(4)["test"][:3]
    gateway = SyntheticCoreGateway(rows)
    context = gateway.payment_context(rows[0]["payment_id"])
    hold = gateway.propose_single_payment_hold(context["payment_order"], context["account"]["core_account_ref"])
    assert hold.scope == "SINGLE_PAYMENT"
    assert hold.approval_required is True
    assert hold.executed is False


def test_metrics_match_hand_calculations() -> None:
    y = [1, 0, 1, 0]
    scores = [0.9, 0.8, 0.7, 0.1]
    assert average_precision(y, scores) == pytest.approx((1 + 2 / 3) / 2)
    assert roc_auc(y, scores) == pytest.approx(0.75)
    assert top_budget(y, scores, 0.5) == {"alerts": 2, "caught": 1, "recall": 0.5, "precision": 0.5}


def test_baselines_return_probabilities() -> None:
    data = generate_dataset(9)
    x, y = matrix(data["train"][:300]), labels(data["train"][:300])
    test = matrix(data["test"][:50])
    for scores in [
        rule_score(test),
        LogisticRegression(iterations=20).fit(x, y).predict_proba(test),
        BoostedStumps(rounds=4).fit(x, y).predict_proba(test),
    ]:
        assert len(scores) == 50
        assert all(0 <= score <= 1 for score in scores)


class FakeResponse:
    def __init__(self, status_code: int = 200, payload: dict | None = None) -> None:
        self.status_code = status_code
        self.payload = payload or {}

    def json(self) -> dict:
        return self.payload


class FakeSession:
    def __init__(self) -> None:
        self.posts: list[tuple[str, dict]] = []
        self.puts: list[bytes] = []

    def post(self, url: str, headers: dict, json: dict, timeout: int) -> FakeResponse:
        self.posts.append((url, json))
        if url.endswith("prepare_train_set_upload"):
            return FakeResponse(
                payload={
                    "train_set_upload_id": "train",
                    "x_train_info": {"signed_urls": ["https://upload/x"], "required_headers": {}},
                    "y_train_info": {"signed_urls": ["https://upload/y"], "required_headers": {}},
                }
            )
        if url.endswith("fit"):
            return FakeResponse(payload={"fitted_train_set_id": "fit"})
        if url.endswith("prepare_test_set_upload"):
            return FakeResponse(
                payload={
                    "test_set_upload_id": "test",
                    "x_test_info": {"signed_urls": ["https://upload/test"], "required_headers": {}},
                }
            )
        return FakeResponse(payload={"prediction": [[0.8, 0.2], [0.1, 0.9]]})

    def put(self, url: str, data: bytes, headers: dict, timeout: int) -> FakeResponse:
        self.puts.append(data)
        return FakeResponse()


def test_rest_client_uses_tabpfn_35_and_returns_positive_probability() -> None:
    session = FakeSession()
    model = TabPFNRestClassifier(token="placeholder", session=session)
    assert model.fit([[1.0, 2.0], [3.0, 4.0]], [0, 1], ["a", "b"]).predict_proba([[1, 2], [3, 4]]) == [0.2, 0.9]
    fit_payload = next(payload for url, payload in session.posts if url.endswith("/fit"))
    assert fit_payload["tabpfn_config"]["model_path"] == "v3.5_default"
    assert session.puts[0].startswith(b"a,b\n")


def test_missing_token_has_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TABPFN_TOKEN", raising=False)
    with pytest.raises(TabPFNAPIError, match="Set TABPFN_TOKEN"):
        TabPFNRestClassifier()


def test_committed_results_are_synthetic_only() -> None:
    demo = json.loads(open("results/demo.json", encoding="utf-8").read())
    assert demo["metrics"]["synthetic_only"] is True
    assert all(row["payment_id"].startswith("TE-") for row in demo["payments"])
