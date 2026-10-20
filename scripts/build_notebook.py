"""Generate notebooks/bank_churn_analysis.ipynb from code cells, then execute it."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = []

cells += [md("""# Bank Customer Churn: risk scoring, survival analysis and fairness check

**Business question.** A retail bank loses about one in five customers. Which customers are most likely to leave, when in their tenure does the risk peak, and where should a limited retention budget go?

**Data.** Public "Churn Modelling" dataset (10,000 customers of a European retail bank; France, Germany, Spain). Target `Exited` = 1 if the customer left the bank. Source and licence are documented in the README.

**Method.** (1) Reproducible pandas + DuckDB pipeline with assertion tests, (2) logistic regression / random forest / XGBoost with class-imbalance handling, stratified 5-fold CV and a 20% hold-out, isotonic calibration, a 0-100 risk score with Low/Medium/High tiers and a retention cost-benefit estimate, (3) Kaplan-Meier and Cox survival analysis on tenure, (4) fairness check by gender and age band.

All numbers below are produced by running this notebook; the same functions live in `src/churn/` and are covered by `tests/`."""),
code("""import sys, warnings
sys.path.insert(0, "../src")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import plotly.io as pio
pio.renderers.default = "png"  # static images so GitHub renders the notebook
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 40); pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
from churn import config as C, data, modeling, survival, fairness, dashboard
print("seed", C.SEED, "| label", C.LABEL, "| features", len(C.FEATURES))"""),

md("""## 1. Data pipeline

Load the raw CSV, validate it, drop identifiers, engineer three features, persist to Parquet and register the table in DuckDB so the same cleaned data can be queried with SQL."""),
code("""raw = data.load_raw()
print(raw.shape)
pd.DataFrame({"column": list(C.DATA_DICTIONARY), "description": list(C.DATA_DICTIONARY.values()), "dtype": [str(raw[c].dtype) for c in C.DATA_DICTIONARY]})"""),
code("""print("Cleaning rules:")
for r in data.CLEANING_RULES: print(" -", r)
df = data.build()
print(df.shape, "| churn rate", round(df[C.LABEL].mean(), 4))
df.head()"""),
code("""# The processed table is also queryable from DuckDB (SQL)
data.sql(\"\"\"
SELECT Geography, NumOfProducts, COUNT(*) AS customers, ROUND(AVG(Exited), 3) AS churn_rate
FROM customers GROUP BY 1, 2 ORDER BY 1, 2
\"\"\")"""),
code("""eda = data.eda_summary(df)
for k in ["Geography", "Gender", "NumOfProducts", "IsActiveMember", "AgeBand"]:
    print(f"\\n== churn rate by {k} =="); print(eda[k].to_string(index=False))"""),

md("""## 2. Classification models

Three model families, each handling the 20% positive rate explicitly (`class_weight='balanced'`, `balanced_subsample`, `scale_pos_weight`). Stratified 5-fold CV on the 8,000-row training split, then a single 2,000-row stratified hold-out that is never touched during model selection."""),
code("""res = modeling.run_all(df)
modeling.save_outputs(res)
print("train", len(res["y_train"]), "| test", len(res["y_test"]))
print("\\nStratified 5-fold CV (train split):")
res["cv_table"].round(3)"""),
code("""print("Hold-out test set, threshold 0.5:")
res["holdout"].round(3)"""),
code("""print("Best model by hold-out ROC AUC:", res["best_name"])
dashboard.fig_roc(res).show()"""),
md("""### Calibration

Weighted models output inflated probabilities. Isotonic calibration with inner 5-fold CV re-maps them so the 0-100 risk score can be read as a probability."""),
code("""print("before:", {k: round(v, 3) for k, v in res["uncalibrated_eval"].items()})
print("after: ", {k: round(v, 3) for k, v in res["calibrated_eval"].items()})
dashboard.fig_calibration(res).show()
res["calibration_after"].round(3)"""),
md("""### Risk score, tiers and retention economics

Score = 100 x calibrated P(churn). Tiers: Low < 20, Medium 20-49, High >= 50. The business case assumes a retention offer costing $50 per targeted customer, $500 of 12-month value kept per retained churner and a 30% offer success rate (all editable in `config.py`)."""),
code("""dashboard.fig_tiers(res).show()
res["tiers"].round(3)"""),
code("""res["economics"].round(1)"""),
code("""best = res["cutoff_sweep"].loc[res["cutoff_sweep"]["net_benefit"].idxmax()]
print(f"Best score cutoff on the hold-out set: {int(best['cutoff'])} -> target {int(best['targeted'])} customers, recall {best['recall']:.0%}, net benefit ${best['net_benefit']:,.0f}")
dashboard.fig_cutoff(res).show()"""),
code("""dashboard.fig_importance(res).show()
res["importance"].head(10)"""),

md("""## 3. Survival analysis

Time = `Tenure` (years with the bank), event = `Exited`. Customers who stayed are right-censored at their current tenure. Kaplan-Meier curves are stratified by Geography, NumOfProducts and IsActiveMember with multivariate log-rank tests; a Cox proportional-hazards model (small L2 penalty) gives adjusted hazard ratios."""),
code("""surv = survival.run_all(df)
survival.save_outputs(surv)
print("log-rank p-values:", {k: f"{v:.2e}" for k, v in surv["logrank_p"].items()})
for col in survival.STRATA:
    dashboard.fig_km(surv, col).show()"""),
code("""print(f"Peak hazard: tenure year {surv['peak_hazard_tenure']} ({surv['peak_hazard']:.1%} of customers reaching that year churn in it)")
dashboard.fig_hazard(surv).show()
surv["hazard"].round(4)"""),
code("""print("Cox PH model | concordance", round(surv["concordance"], 4))
dashboard.fig_cox(surv).show()
surv["cox_table"].round(4)"""),
code("""print("Proportional-hazards assumption test (p < 0.05 would flag a violation):")
surv["ph_test"].round(4)"""),

md("""## 4. Fairness check

On the hold-out set, compare recall (true positive rate), false positive rate, precision and selection rate across Gender and age bands at the High-tier cutoff (score >= 50)."""),
code("""fair = fairness.run_all(res["X_test"], res["y_test"], res["score"])
fairness.save_outputs(fair)
print("Gender gaps:", {k: round(v, 3) for k, v in fair["gender_gaps"].items()})
dashboard.fig_fairness(fair, "by_gender", "Fairness by Gender at score >= 50").show()
fair["by_gender"].round(3)"""),
code("""print("Age-band gaps:", {k: round(v, 3) for k, v in fair["age_gaps"].items()})
dashboard.fig_fairness(fair, "by_age", "Fairness by age band at score >= 50").show()
fair["by_age"].round(3)"""),
md("""**Reading.** Gender gaps are small (recall and false-positive-rate differences of a few points, both tracking the higher base rate among women). Age gaps are large: churn is concentrated in the 41-60 bands and Age is a top feature, so the model recalls most churners there and almost none under 30. A High-tier threshold therefore under-serves young churners; if the retention programme must reach them, use per-band thresholds or a separate young-customer campaign rather than a single global cutoff. See the model card for the full discussion."""),

md("""## 5. Dashboard export

Writes `reports/dashboard.html`, a self-contained plotly page with every chart above plus table views."""),
code("""path = dashboard.build(res, surv, fair)
print("dashboard written to", path)
pngs = dashboard.save_figures(res, surv, fair)
print(len(pngs), "PNG figures written to reports/figures/")"""),
]
nb["cells"] = cells
nb.metadata["kernelspec"] = {"name": "bank-churn-risk", "display_name": "Python (bank-churn-risk)", "language": "python"}
out = Path("notebooks/bank_churn_analysis.ipynb")
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print("wrote", out)
