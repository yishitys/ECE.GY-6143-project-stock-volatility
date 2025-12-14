# Stock Volatility Prediction — Professor Report

**Project**: Reddit discussion volume + text embeddings for short-horizon volatility prediction  
**Symbol**: GME  
**Target**: `target_volatility_log_return_abs` (next-hour volatility proxy)  
**Generated**: 2025-12-13 22:47:07

---

## 1. Data Summary

- **Rows**: 2001
- **Columns**: 444
- **Time range**: 2021-01-04 14:00:00+00:00 → 2021-12-30 20:00:00+00:00
- **Text embedding features**: 384 (`embedding_0..`)
- **Reddit volume features**: 4 (post_count, total_comments, total_score, unique_authors)

## 2. Experimental Protocol (Time-Series Split)

We use a strict chronological split (no shuffle):
- **Train**: 70% — 2021-01-04 14:00:00+00:00 → 2021-09-14 18:00:00+00:00
- **Val**: 15% — 2021-09-14 19:00:00+00:00 → 2021-11-05 14:00:00+00:00
- **Test**: 15% — 2021-11-05 15:00:00+00:00 → 2021-12-30 20:00:00+00:00

For fair comparison with the LSTM sequence model, we **exclude the first 24 test points** (sequence warm-up) when computing test metrics for *all* models.

## 3. Models / Baselines

- **Baseline-Mean**: predict a constant mean value (simple sanity check)
- **Baseline-Persistence**: predict the next value as the previous observed value
- **XGBoost-full**: all features (technical + Reddit volume + embeddings)
- **XGBoost-no_text**: remove `embedding_*` (keeps Reddit volume + technical)
- **XGBoost-tech_only**: remove Reddit volume + embeddings (technical/price only)
- **LSTM**: sequence model over all numeric features (standardized)

## 4. Metrics (Test)

We report: RMSE / MAE / R² / MAPE / Directional Accuracy / Correlation.

| model                | split   |     rmse |      mae |      r2 |        mape |   directional_accuracy |   correlation |
|:---------------------|:--------|---------:|---------:|--------:|------------:|-----------------------:|--------------:|
| XGBoost-full         | test    | 0.015983 | 0.007069 |  0.3014 | 3.26144e+07 |                52.5362 |        0.5782 |
| XGBoost-no_text      | test    | 0.016332 | 0.008081 |  0.2706 | 4.3581e+07  |                85.1449 |        0.6387 |
| XGBoost-tech_only    | test    | 0.018381 | 0.009149 |  0.0761 | 5.32899e+07 |                73.913  |        0.2949 |
| LSTM                 | test    | 0.018965 | 0.008553 |  0.0165 | 4.13474e+07 |                15.942  |        0.1313 |
| Baseline-Mean        | test    | 0.020041 | 0.013542 | -0.0983 | 9.66825e+07 |                75.3623 |        0      |
| Baseline-Persistence | test    | 0.027962 | 0.01005  | -1.1381 | 5.02498e+07 |                63.0435 |       -0.069  |

## 5. Figures

All figures are saved under `results/figures/`:

- Predictions vs Actual:
  - `baseline_mean_predictions.png`
  - `baseline_persistence_predictions.png`
  - `xgboost_full_predictions.png`
  - `xgboost_no_text_predictions.png`
  - `xgboost_tech_only_predictions.png`
  - `lstm_predictions.png`
- Residual diagnostics:
  - `*_residuals.png`
- Error distributions:
  - `*_errors.png`
- XGBoost feature importance:
  - `xgboost_full_importance.png`
  - `xgboost_no_text_importance.png`
  - `xgboost_tech_only_importance.png`

## 6. Reproducibility

Run this report generator from repo root:

```bash
python ECE.GY-6143-project-stock-volatility/src/generate_professor_report.py --symbol GME
```
