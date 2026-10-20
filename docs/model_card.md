# Model card: bank customer churn risk score

## Model details

- **Model**: XGBoost classifier (400 trees, depth 4, learning rate 0.03,
  `scale_pos_weight` = 3.9) wrapped in isotonic calibration (5-fold inner CV),
  selected over logistic regression and random forest by hold-out ROC AUC.
- **Output**: calibrated probability of churn, reported as a 0-100 risk score
  with tiers Low (< 20), Medium (20-49), High (>= 50).
- **Inputs (13)**: CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard,
  IsActiveMember, EstimatedSalary, BalanceSalaryRatio, ZeroBalance,
  ProductsPerTenureYear, Geography, Gender.
- **Training data**: 8,000 customers (stratified 80% of the public "Churn
  Modelling" dataset); evaluation on the remaining 2,000. Seed 42.
- **Version**: trained by `notebooks/bank_churn_analysis.ipynb`; metrics in
  `reports/metrics.json`.
- **Author / contact**: Chenhan Gao, 2026. Personal portfolio project.

## Intended use

- Rank existing retail-bank customers by likelihood of leaving so a retention
  team can prioritise outreach and size a retention budget.
- Companion survival analysis (Kaplan-Meier, Cox) tells the team *when* in the
  tenure the risk concentrates (years 9-10 in this data).

**Out of scope**: credit or pricing decisions; any use on a population that
differs from the training data (other countries, business customers); any use
without a review of the fairness section below.

## Performance (hold-out, n = 2,000, base rate 20.4%)

| Metric | Value |
|---|---|
| ROC AUC | 0.868 |
| PR AUC | 0.720 |
| Brier score | 0.099 (0.136 before calibration) |
| Precision / recall at score >= 50 | 0.80 / 0.48 |
| Precision / recall at score >= 40 (net-benefit optimum) | 0.73 / 0.57 |
| High tier (12.7% of customers) | 78.7% churn rate, 48.9% of all churners, 3.9x lift |

Cross-validated ROC AUC on the training split: 0.865 +/- 0.010 (5 folds).
Calibration: decile-binned predicted vs observed rates lie within about 3 points
across the range after isotonic calibration (table in the notebook).

## Fairness and ethical considerations

Evaluated at the High-tier cutoff (score >= 50) on the hold-out set.

| Dimension | Recall gap (max - min) | FPR gap | Selection-rate ratio (min/max) |
|---|---|---|---|
| Gender | 5.1 points (F 51.1%, M 46.0%) | 1.6 points (F 4.3%, M 2.7%) | 0.61 |
| Age band | 67.3 points (18-30: 8.0%, 51-60: 75.3%) | 11.1 points | 0.01 |

- **Gender**: small gaps, both tracking the higher base rate among women
  (25.1% vs 16.5%). Gender is a model input with a Cox hazard ratio of 0.72 for
  men. Whether a protected attribute may influence retention offers is a policy
  decision; removing it costs little accuracy and should be tested before any
  real deployment.
- **Age**: large gaps. Churn is concentrated at ages 41-60 and Age is the
  second most important feature, so a single global cutoff recalls 75% of
  churners aged 51-60 but only 8% of churners under 30. This is an accurate
  description of the data, not a modelling error, but it means the score would
  systematically under-serve young customers in a retention programme.
  Mitigations: per-age-band cutoffs chosen on net benefit, or a separate
  young-customer campaign; re-run `fairness.run_all` under the chosen policy.
- **Proxies**: `Surname` was dropped as a possible ethnicity proxy; `Geography`
  is kept because it is a legitimate market segment, but note that Germany has a
  hazard ratio of 1.71.

## Limitations

- Cross-sectional snapshot, no time stamps: no out-of-time validation and no
  behavioural signals. The survival analysis treats Tenure as time-to-event with
  right-censoring, which assumes cohorts are comparable.
- Small (10,000 rows) and partly synthetic-looking public data; results show the
  method, not a real bank.
- Economic assumptions ($50 offer, $500 retained value, 30% success) are
  illustrative and should be replaced with the bank's own numbers.
- Model has not been monitored for drift; re-train and re-check fairness on
  each refresh.

## Reproducibility

`python -m pytest -q` (16 tests), then execute the notebook. All numbers in this
card come from `reports/metrics.json`, `reports/fairness_by_*.csv` and
`reports/cox_hazard_ratios.csv` produced by that run.
