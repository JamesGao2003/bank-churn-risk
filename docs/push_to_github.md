# Push to GitHub (run these yourself)

```bash
cd projects/bank-churn-risk
git init -b main
git add .
git commit -m "Bank customer churn risk scoring, survival analysis and fairness check"
gh repo create JamesGao2003/bank-churn-risk --public --source=. --remote=origin --push
```

Without the GitHub CLI: create an empty public repository named `bank-churn-risk`
at github.com/new, then

```bash
git remote add origin https://github.com/JamesGao2003/bank-churn-risk.git
git push -u origin main
```

`.gitignore` already excludes `.venv/`, caches and the DuckDB file; the
executed notebook (about 0.7 MB with static PNG outputs), the raw CSV (0.7 MB),
the dashboard and the figures are committed on purpose so the results render on
GitHub without running anything.
