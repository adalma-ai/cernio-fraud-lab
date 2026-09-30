# Model card

## Intended use

This project demonstrates TabPFN-3.5 for ranking synthetic pre-release payments in a fraud review queue. It is a research and demo artifact, not a production fraud control.

## Data

The generator creates independent synthetic train, validation, and later-period test sets. Scam patterns cover romance or investment scams, gift card scams, tech support scams, account takeover, and benefits reroutes. Legitimate hard negatives include home repairs, family support, travel bookings, and new payroll payees.

No customer, member, bank, or production data is included.

## Inputs

The model uses 10 numeric features available before payment release. The target label and scam type are excluded.

## Outputs

The model returns a scam probability. The demo converts that score into queue rank, a fraud evaluation record, and, for the top 5%, a proposed single-payment hold and fraud review case.

## Evaluation

Primary metrics are average precision, recall at a fixed review budget, precision at a fixed review budget, and alerts needed to reach 80% recall. Brier score is reported for probability quality. Thresholds are selected on validation data and evaluated on the untouched later-period test data.

## Human oversight

Every hold is a recommendation that requires analyst approval. The public code has no action executor.

## Known risks

- Synthetic patterns may be easier than real fraud.
- Scam tactics change over time.
- Features such as device or contact changes can be missing or delayed in real systems.
- A false positive can delay a legitimate payment. Any production control needs escalation, customer communication, and a release path.
