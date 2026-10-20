# Resume bullets (Projects section)

**Bank Customer Churn Risk Scoring | Python, SQL (DuckDB), scikit-learn, XGBoost, lifelines, Plotly** | github.com/JamesGao2003/bank-churn-risk

- Built a tested, reproducible churn pipeline (pandas, DuckDB, 16 pytest checks) on 10,000 retail-bank customers; compared logistic regression, random forest and XGBoost with stratified 5-fold CV and a hold-out set, reaching ROC AUC 0.87 / PR AUC 0.72 and cutting Brier score 27% (0.136 to 0.099) with isotonic calibration.
- Converted calibrated probabilities into a 0-100 risk score with Low/Medium/High tiers; the High tier (13% of customers) captured 49% of churners at 3.9x lift, and a cost-benefit sweep identified a cutoff worth $18.7K net per 2,000 customers versus a $39K loss from untargeted outreach.
- Ran Kaplan-Meier and Cox survival analysis on tenure (hazard ratios: 3+ products 2.6x, Germany 1.7x, active members 0.54x, all p < 0.001; churn hazard peaks in years 9-10) and audited the score by gender and age band (5-point recall gap by gender, 67-point gap by age), documented in a model card and interactive dashboard.

Each bullet fits in two lines at Calibri 10pt on a one-page resume; trim the parenthetical numbers first if a line wraps.
