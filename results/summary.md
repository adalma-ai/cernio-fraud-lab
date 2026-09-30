# Results

Synthetic data only. Test sets are generated independently as a later period.

| Model | AP | ROC AUC | Recall at 5% review | Precision at 5% review | Alerts for 80% recall | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TabPFN-3.5 | 0.961 +/- 0.006 | 0.996 | 81.1% | 97.3% | 49 | 0.0091 |
| Boosted stumps | 0.765 +/- 0.061 | 0.963 | 62.2% | 74.7% | 99 | 0.0343 |
| Logistic regression | 0.792 +/- 0.048 | 0.977 | 63.3% | 76.0% | 87 | 0.0263 |
| Rule score | 0.516 +/- 0.082 | 0.941 | 42.8% | 51.3% | 121 | 0.0428 |

## Training-size curve

| Train rows | Frauds | Model | AP | Recall at 5% review |
| ---: | ---: | --- | ---: | ---: |
| 150 | 12 | TabPFN-3.5 | 0.732 | 56.7% |
| 150 | 12 | Boosted stumps | 0.602 | 45.0% |
| 150 | 12 | Logistic regression | 0.663 | 51.7% |
| 150 | 12 | Rule score | 0.465 | 38.3% |
| 300 | 24 | TabPFN-3.5 | 0.885 | 71.7% |
| 300 | 24 | Boosted stumps | 0.705 | 55.0% |
| 300 | 24 | Logistic regression | 0.685 | 55.0% |
| 300 | 24 | Rule score | 0.465 | 38.3% |
| 600 | 48 | TabPFN-3.5 | 0.949 | 78.3% |
| 600 | 48 | Boosted stumps | 0.639 | 45.0% |
| 600 | 48 | Logistic regression | 0.705 | 58.3% |
| 600 | 48 | Rule score | 0.465 | 38.3% |
| 1200 | 96 | TabPFN-3.5 | 0.967 | 81.7% |
| 1200 | 96 | Boosted stumps | 0.704 | 60.0% |
| 1200 | 96 | Logistic regression | 0.739 | 61.7% |
| 1200 | 96 | Rule score | 0.465 | 38.3% |
