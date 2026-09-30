# Plan

## Goal

Build a standalone TabPFN-3.5 hackathon entry that asks one operational
question: when a fraud team can review only a small share of outgoing payments,
which payments should enter the queue first?

## Scope

- Fresh synthetic generator for scam and legitimate payments.
- Features restricted to evidence visible before a candidate payment is released.
- Separate train, validation, and later-period test sets.
- Hosted TabPFN-3.5 through the official REST API.
- Pure-Python baselines: a transparent rule score, logistic regression, and
  boosted decision stumps.
- Metrics for ranking, calibration, review workload, and per-typology recall.
- Static local dashboard with review budget controls and an evidence queue.
- Tests for data generation, leakage checks, metrics, baselines, and REST flow.

## Out of scope

- Customer data, private data, or arbitrary CSV uploads.
- Production integration with the Cernio monorepo.
- Claims about real-world fraud performance.

## Verification

1. Run unit tests offline.
2. Run the full experiment with the hosted API.
3. Inspect the dashboard in a browser.
4. Scan the project for secrets and generated credentials before publication.
