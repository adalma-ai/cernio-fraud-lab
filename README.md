# Cernio Fraud Lab

Cernio Fraud Lab uses TabPFN-3.5 to rank synthetic outgoing payments for fraud review before release. The question is practical: when an analyst team can review only a small share of payments, which ones should it see first?

The project combines three parts:

- a deterministic synthetic scam-payment generator;
- BIAN-aligned payment, party, account, fraud-evaluation, and case records;
- a local analyst queue that compares TabPFN-3.5 with simple baselines at different review budgets.

All data is synthetic. The hosted API run sends synthetic rows to Prior Labs. Do not use this project with customer or production data.

## Result

Test data is generated independently as a later period. Each run trains on 1,200 synthetic payments, selects thresholds on 500 validation payments, and evaluates 1,000 test payments. The table below is the mean across seeds 7, 11, and 19.

| Model | Average precision | ROC AUC | Recall at 5% review | Precision at 5% review | Alerts for 80% recall | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TabPFN-3.5 | 0.961 ± 0.006 | 0.996 | 81.1% | 97.3% | 49 | 0.0091 |
| Logistic regression | 0.792 ± 0.048 | 0.977 | 63.3% | 76.0% | 87 | 0.0263 |
| Boosted stumps | 0.765 ± 0.061 | 0.963 | 62.2% | 74.7% | 99 | 0.0343 |
| Rule score | 0.516 ± 0.082 | 0.941 | 42.8% | 51.3% | 121 | 0.0428 |

With a 5% review budget, TabPFN-3.5 finds 81.1% of synthetic scams. The best baseline finds 63.3%. Reaching 80% recall takes 49 alerts with TabPFN-3.5, compared with 87 for logistic regression.

TabPFN-3.5 also leads in the training-size check for seed 7:

| Training rows | Scam rows | TabPFN-3.5 AP | Best baseline AP |
| ---: | ---: | ---: | ---: |
| 150 | 12 | 0.732 | 0.663 |
| 300 | 24 | 0.885 | 0.705 |
| 600 | 48 | 0.949 | 0.705 |
| 1,200 | 96 | 0.967 | 0.739 |

Full results are in [`results/summary.md`](results/summary.md) and [`results/metrics.json`](results/metrics.json).

## Run it

Requirements: Python 3.11 or newer and a Prior Labs API key.

```bash
python3 -m pip install -r requirements.txt
cp .env.example .env
# Put your key in .env as TABPFN_TOKEN=...
set -a; source .env; set +a
python3 -m cernio_fraud_lab.experiment --output results
python3 -m pytest -q
python3 -m http.server 8000
```

Open <http://localhost:8000/app/>.

Offline baseline check, with no API calls:

```bash
python3 -m cernio_fraud_lab.experiment --offline --seeds 7 --no-curve --output /tmp/cernio-fraud-lab
```

## How TabPFN-3.5 is used

`cernio_fraud_lab/tabpfn_api.py` calls the Prior Labs REST API with `model_path="v3.5_default"`. It uploads the training table, fits once, uploads validation and test rows together, and receives class probabilities.

TabPFN receives only these pre-release features:

| Feature | Meaning |
| --- | --- |
| `amount_to_p90_ratio` | Payment amount divided by the account's usual high outbound payment |
| `payee_age_days` | Days since the payee was added |
| `first_time_payee` | First payment to this payee |
| `outbound_count_24h` | Outgoing payments in the prior 24 hours |
| `hour_local` | Local hour when the payment was initiated |
| `rail_code` | Payment rail and beneficiary category |
| `balance_drawdown_pct` | Share of balance leaving the account |
| `days_since_contact_change` | Days since phone or email changed |
| `new_device` | Session came from a new device |
| `dormant_days_before` | Days since earlier account activity |

The label and scam type never enter the model input.

## Canonical data model

The public model follows the shape of a BIAN payment flow:

| Record | BIAN business object |
| --- | --- |
| `PaymentOrder` | Payments.PaymentOrder.PaymentOrderInstruction |
| `PaymentParty` | Payment originator and beneficiary |
| `Party` reference | Party.PartyReferenceDataDirectory.PartyReference |
| `CustomerCase` | CustomerCase.CustomerCaseManagement.Case |

Behavioral features, model scores, the core account reference, and hold proposals are marked as `cernio_lab.*` extensions. The generated features are rebuilt from canonical payment and behavior records, and a test checks that path.

## Core gateway boundary

`cernio_fraud_lab/gateway.py` defines a vendor-neutral System API boundary:

- `payment_context(payment_order_id)` reads the canonical payment, account, and behavior context;
- `propose_single_payment_hold(order, account_ref)` returns a single-payment hold proposal.

The public demo includes only an in-memory synthetic gateway. It never places, releases, or cancels a hold. A production adapter for a core connectivity provider belongs in a private integration layer with its own credentials, contract tests, and audit trail.

## Architecture

```text
synthetic generator
  -> PaymentOrder + AccountReference + BehaviorContext
  -> feature extraction
  -> TabPFN-3.5 and baselines
  -> FraudEvaluation and review queue
  -> CustomerCase + SINGLE_PAYMENT hold proposal
```

## Baselines

The baselines are written in plain Python, so they run in restricted environments without native packages:

- a fixed analyst rule score;
- L2-regularized logistic regression with rail one-hot encoding and cyclical time features;
- boosted decision stumps.

They are useful controls, but they are not tuned XGBoost or LightGBM models.

## Limitations

- The data is synthetic and was designed with separable scam patterns. These numbers do not predict production performance.
- The model has no network graph, device fingerprint history, text, or feedback loop.
- The evidence chips are observable facts, not causal or model explanations.
- The demo does not connect to a core system and does not execute holds.
- The hosted API is suitable for this synthetic benchmark only. A regulated production deployment needs data-residency, security, and model-risk review.

See [`MODEL_CARD.md`](MODEL_CARD.md).

## License

Apache License 2.0.
