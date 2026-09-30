"""Run the synthetic analyst-queue benchmark."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any, Callable

from . import __version__
from .baselines import BoostedStumps, LogisticRegression, rule_score
from .canonical import BIAN_FIELD_MAP, EXTENSIONS, CustomerCase, FraudEvaluation, to_dict
from .gateway import SyntheticCoreGateway
from .metrics import best_f1_threshold, summarize, top_budget
from .synthetic import FEATURES, RAILS, generate_dataset, labels, matrix
from .tabpfn_api import TabPFNRestClassifier

MODEL_ORDER = ["TabPFN-3.5", "Boosted stumps", "Logistic regression", "Rule score"]


def _stratified_prefix(rows: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    frauds = [row for row in rows if row["label"]]
    legit = [row for row in rows if not row["label"]]
    fraud_count = max(3, round(size * len(frauds) / len(rows)))
    return frauds[:fraud_count] + legit[: size - fraud_count]


def _score_models(
    train: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    use_tabpfn: bool,
) -> tuple[dict[str, list[float]], dict[str, float]]:
    x_train, y_train, x_score = matrix(train), labels(train), matrix(score_rows)
    factories: dict[str, Callable[[], list[float]]] = {
        "Boosted stumps": lambda: BoostedStumps().fit(x_train, y_train).predict_proba(x_score),
        "Logistic regression": lambda: LogisticRegression().fit(x_train, y_train).predict_proba(x_score),
        "Rule score": lambda: rule_score(x_score),
    }
    if use_tabpfn:
        factories = {
            "TabPFN-3.5": lambda: TabPFNRestClassifier().fit(x_train, y_train, FEATURES).predict_proba(x_score),
            **factories,
        }
    scores: dict[str, list[float]] = {}
    timings: dict[str, float] = {}
    for name, factory in factories.items():
        start = time.perf_counter()
        scores[name] = factory()
        timings[name] = time.perf_counter() - start
    return scores, timings


def _aggregate(runs: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    names = [name for name in MODEL_ORDER if name in runs[0]["test"]]
    aggregate: dict[str, dict[str, float]] = {}
    for name in names:
        values = {
            "average_precision": [run["test"][name]["average_precision"] for run in runs],
            "roc_auc": [run["test"][name]["roc_auc"] for run in runs],
            "brier": [run["test"][name]["brier"] for run in runs],
            "recall_top_5pct": [run["test"][name]["top_5pct"]["recall"] for run in runs],
            "precision_top_5pct": [run["test"][name]["top_5pct"]["precision"] for run in runs],
            "alerts_for_80pct_recall": [run["test"][name]["alerts_for_80pct_recall"] for run in runs],
        }
        aggregate[name] = {}
        for metric, series in values.items():
            aggregate[name][f"{metric}_mean"] = statistics.mean(series)
            aggregate[name][f"{metric}_std"] = statistics.stdev(series) if len(series) > 1 else 0.0
    return aggregate


def _evidence(row: dict[str, Any]) -> list[str]:
    f = row["features"]
    evidence = []
    if f["amount_to_p90_ratio"] >= 2:
        evidence.append(f"{f['amount_to_p90_ratio']:.1f}x usual high payment")
    if f["first_time_payee"]:
        evidence.append("first payment to payee")
    if f["payee_age_days"] < 14:
        evidence.append(f"payee added {f['payee_age_days']:.0f} days ago")
    if f["new_device"]:
        evidence.append("new device")
    if f["days_since_contact_change"] < 7:
        evidence.append("contact details changed this week")
    if f["outbound_count_24h"] >= 3:
        evidence.append(f"{f['outbound_count_24h']:.0f} outbound payments in 24h")
    if f["rail_code"] in {2.0, 4.0, 5.0}:
        evidence.append(RAILS[int(f["rail_code"])])
    return evidence[:5]


def run(seeds: list[int], output: Path, use_tabpfn: bool = True, learning_curve: bool = True) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    runs = []
    demo_rows: list[dict[str, Any]] = []
    for seed in seeds:
        data = generate_dataset(seed)
        validation, test = data["validation"], data["test"]
        scores, timings = _score_models(data["train"], validation + test, use_tabpfn)
        val_scores = {name: values[: len(validation)] for name, values in scores.items()}
        test_scores = {name: values[len(validation) :] for name, values in scores.items()}
        thresholds = {name: best_f1_threshold(labels(validation), values) for name, values in val_scores.items()}
        runs.append(
            {
                "seed": seed,
                "counts": {name: {"rows": len(rows), "frauds": sum(labels(rows))} for name, rows in data.items()},
                "timings_seconds": timings,
                "test": {name: summarize(test, test_scores[name], thresholds[name]) for name in test_scores},
            }
        )
        if seed == seeds[0]:
            primary = "TabPFN-3.5" if "TabPFN-3.5" in test_scores else "Boosted stumps"
            order = sorted(range(len(test)), key=lambda index: test_scores[primary][index], reverse=True)
            ranks = {index: rank for rank, index in enumerate(order, 1)}
            gateway = SyntheticCoreGateway(test)
            for i, row in enumerate(test):
                context = gateway.payment_context(row["payment_id"])
                evidence = _evidence(row)
                evaluation = FraudEvaluation(
                    evaluation_id=f"FE-{row['payment_id']}",
                    payment_order_id=row["payment_id"],
                    model=primary,
                    score=round(test_scores[primary][i], 6),
                    queue_rank=ranks[i],
                    evidence=evidence,
                )
                action: dict[str, Any] = {}
                if ranks[i] <= max(1, len(test) // 20):
                    hold = gateway.propose_single_payment_hold(
                        context["payment_order"], context["account"]["core_account_ref"]
                    )
                    case = CustomerCase(
                        case_id=f"CASE-{row['payment_id']}",
                        case_type="CASE_TYPE_FRAUD_REVIEW",
                        status="CASE_STATUS_OPEN",
                        severity="CASE_SEVERITY_HIGH" if ranks[i] <= 10 else "CASE_SEVERITY_MEDIUM",
                        subject_party_id=context["account"]["party_id"],
                        narrative="Synthetic payment queued for analyst review. Evidence: " + "; ".join(evidence),
                    )
                    action = {"hold_recommendation": to_dict(hold), "customer_case": to_dict(case)}
                demo_rows.append(
                    {
                        "payment_id": row["payment_id"],
                        "label": row["label"],
                        "typology": row["typology"],
                        "features": row["features"],
                        "evidence": evidence,
                        "scores": {name: round(values[i], 6) for name, values in test_scores.items()},
                        "canonical": context,
                        "fraud_evaluation": to_dict(evaluation),
                        **action,
                    }
                )

    curve = []
    if learning_curve:
        data = generate_dataset(seeds[0])
        test = data["test"]
        for size in [150, 300, 600, 1200]:
            train = _stratified_prefix(data["train"], size)
            curve_scores, _ = _score_models(train, test, use_tabpfn)
            curve.append(
                {
                    "train_rows": len(train),
                    "train_frauds": sum(labels(train)),
                    "models": {
                        name: {
                            "average_precision": summarize(test, values, 0.5)["average_precision"],
                            "recall_top_5pct": top_budget(labels(test), values, 0.05)["recall"],
                        }
                        for name, values in curve_scores.items()
                    },
                }
            )

    result = {
        "project": "Cernio Fraud Lab",
        "version": __version__,
        "synthetic_only": True,
        "model_endpoint": "Prior Labs hosted API, model_path=v3.5_default" if use_tabpfn else "offline baselines only",
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "features": FEATURES,
        "bian_field_map": BIAN_FIELD_MAP,
        "extensions": EXTENSIONS,
        "seeds": seeds,
        "aggregate_test": _aggregate(runs),
        "runs": runs,
        "learning_curve": curve,
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "demo.json").write_text(
        json.dumps({"metrics": result, "payments": demo_rows}, separators=(",", ":")) + "\n"
    )
    (output / "summary.md").write_text(render_summary(result))
    return result


def render_summary(result: dict[str, Any]) -> str:
    lines = [
        "# Results",
        "",
        "Synthetic data only. Test sets are generated independently as a later period.",
        "",
        "| Model | AP | ROC AUC | Recall at 5% review | Precision at 5% review | Alerts for 80% recall | Brier |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, values in result["aggregate_test"].items():
        lines.append(
            f"| {name} | {values['average_precision_mean']:.3f} +/- {values['average_precision_std']:.3f} | "
            f"{values['roc_auc_mean']:.3f} | {values['recall_top_5pct_mean']:.1%} | "
            f"{values['precision_top_5pct_mean']:.1%} | {values['alerts_for_80pct_recall_mean']:.0f} | "
            f"{values['brier_mean']:.4f} |"
        )
    if result["learning_curve"]:
        lines += ["", "## Training-size curve", "", "| Train rows | Frauds | Model | AP | Recall at 5% review |", "| ---: | ---: | --- | ---: | ---: |"]
        for point in result["learning_curve"]:
            for name, values in point["models"].items():
                lines.append(
                    f"| {point['train_rows']} | {point['train_frauds']} | {name} | "
                    f"{values['average_precision']:.3f} | {values['recall_top_5pct']:.1%} |"
                )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 19])
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--offline", action="store_true", help="Run pure-Python baselines only.")
    parser.add_argument("--no-curve", action="store_true", help="Skip the training-size curve.")
    args = parser.parse_args()
    result = run(args.seeds, args.output, use_tabpfn=not args.offline, learning_curve=not args.no_curve)
    print(render_summary(result))


if __name__ == "__main__":
    main()
