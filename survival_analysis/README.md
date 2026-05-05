# Homework 3: Survival Analysis

## NOTE: The report is in the outputs directory

This repository implements the full DS 223 Homework 3 survival-analysis workflow for the `telco.csv` dataset.

## What is included

- Parametric AFT models:
  - Exponential
  - Weibull
  - Log-normal
  - Log-logistic
- Model comparison using log-likelihood, AIC, and BIC.
- One combined survival-curve plot for all AFT distributions.
- Significant-feature selection from the best full model.
- Final reduced AFT model.
- Per-customer CLV calculation over the selected month horizon.
- CLV exploration by customer segments.
- Annual retention budget estimate based on at-risk customers.
- Short written report.

## Main files

```text
data/telco.csv
src/homework3_survival.py
homework3_survival_analysis.ipynb
outputs/report.md
outputs/model_comparison_with_final.csv
outputs/final_model_coefficients.csv
outputs/clv_per_customer.csv
outputs/segment_summary.csv
outputs/annual_retention_budget.csv
outputs/plots/
```

## Output file guide

- `outputs/report.md`: short 1-2 paragraph interpretation for the homework prompt.
- `outputs/model_comparison_full.csv`: AIC/BIC comparison for the full Exponential, Weibull, Log-normal, and Log-logistic AFT models.
- `outputs/model_comparison_with_final.csv`: same comparison plus the reduced final model used for interpretation.
- `outputs/best_full_model_coefficients.csv`: coefficient table for the best full model before feature reduction.
- `outputs/final_model_coefficients.csv`: coefficient table for the reduced final model; use this to interpret churn-risk factors.
- `outputs/clv_per_customer.csv`: customer-level survival probabilities, churn risk, at-risk flag, and CLV.
- `outputs/segment_summary.csv`: average CLV, survival, churn risk, and at-risk counts by segment.
- `outputs/annual_retention_budget.csv`: at-risk subscriber count, expected churners, CLV at risk, and suggested annual retention budget.
- `outputs/scaler.json`: means and standard deviations used to standardize numeric variables.
- `outputs/plots/aft_survival_curves_all_models.png`: survival curves for the fitted AFT distributions.
- `outputs/plots/clv_distribution.png`: distribution of customer CLV.
- `outputs/plots/clv_by_custcat.png`: CLV comparison by customer category.
- `outputs/plots/churn_risk_by_custcat.png`: average 12-month churn risk by customer category.

## How to run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full analysis:

```bash
python src/homework3_survival.py --data data/telco.csv --output-dir outputs
```

Optional parameters:

```bash
python homework3_survival.py \
  --data data/telco.csv \
  --output-dir outputs \
  --monthly-margin 1300 \
  --annual-discount-rate 0.10 \
  --clv-months 12 \
  --at-risk-threshold 0.25
```

## Interpretation note

The final model is an Accelerated Failure Time model. Positive coefficients increase predicted survival time and reduce churn risk over the same horizon. Negative coefficients shorten predicted survival time and increase churn risk. Numeric variables are standardized, so their coefficients are interpreted per one standard deviation increase.
