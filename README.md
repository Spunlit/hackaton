# Higgsfield Retention Architect — AI development jiggas department

**HackNU 2026 | Higgsfield DS Challenge**

## Task

Build a predictive system that identifies users at risk of leaving the platform **within the first 14 days** of their subscription and classifies the churn type:

| Label | Meaning |
|---|---|
| `not_churned` | User renewed subscription |
| `vol_churn` | User actively cancelled |
| `invol_churn` | Subscription cancelled due to payment failure |

**Evaluation metric:** Weighted F1-score

## Results

| Version | OOF Weighted F1 |
|---|---|
| Baseline (LightGBM multiclass) | 0.5686 |
| + Time-based features | 0.5855 |
| + Target encoding + Cascade | 0.6011 |
| + Alpha sweep optimization | 0.6129 |
| **Final (GPU + threshold tuning)** | **0.6272** |

## Solution Architecture

```
                    ┌─────────────────────────────┐
                    │     Feature Engineering      │
                    │  (130 features, cached)      │
                    └────────────┬────────────────┘
                                 │
              ┌──────────────────┴──────────────────┐
              │                                     │
    ┌─────────▼──────────┐              ┌───────────▼──────────┐
    │     STAGE 1        │              │    LightGBM Multi    │
    │  Churn vs No-Churn │              │    (3-class GPU)     │
    │  (Binary, GPU)     │              │    OOF F1: 0.5916    │
    └─────────┬──────────┘              └───────────┬──────────┘
              │                                     │
    ┌─────────▼──────────┐                          │
    │     STAGE 2        │                          │
    │   Vol vs Invol     │                          │
    │  (Binary, GPU)     │                          │
    └─────────┬──────────┘                          │
              │                                     │
              └──────────────┬──────────────────────┘
                             │
              ┌──────────────▼──────────────────────┐
              │     Ensemble + Threshold Sweep       │
              │   cascade(0.8) + lgbm(0.2)           │
              │   + Stage1 threshold optimization    │
              │   + Vol class scale sweep            │
              └──────────────┬──────────────────────┘
                             │
                    OOF F1 = 0.6272
```

## Feature Groups (130 total)

| Group | Count | Key Signals |
|---|---|---|
| Generation behavior | ~50 | span_hours, activity_slope, per_active_day, first_latency |
| Target encoding | 21 | Country × churn_rate, plan × churn_rate, source × churn_rate |
| Transaction / payment | 25 | is_prepaid, fail_rate, card_declined, country_mismatch |
| Onboarding quiz | 15 | frustration, role, usage_plan, first_feature |
| Subscription properties | 7 | plan_tier, start_day, country |
| Composite signals | 5 | invol_risk, vol_risk, engagement, plan×engagement |

## How to Run

```bash
# First run: builds features (~6 min) + trains models (~12 min)
python train.py

# Subsequent runs: loads from cache, trains only (~12 min)
python train.py
```

**Requirements:** Python 3.10+, LightGBM 4.x with GPU, CUDA-compatible GPU

```bash
pip install lightgbm xgboost scikit-learn pandas numpy scipy pyarrow
```

## File Structure

```
Hackaton2/
├── train.py                                    # Full pipeline
├── feature_importance.csv                      # Feature importance from LightGBM
├── AI_development_jiggas_department_submission.csv  # Final submission
├── feature_cache/
│   ├── train_features.parquet                  # Cached train features
│   └── test_features.parquet                   # Cached test features
├── Train Data/                                 # Raw train datasets
└── Test Data/                                  # Raw test datasets
```
