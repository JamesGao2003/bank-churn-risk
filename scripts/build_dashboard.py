import sys; sys.path.insert(0, "src")
from churn import data, modeling, survival, fairness, dashboard
df = data.load_clean(); res = modeling.run_all(df); surv = survival.run_all(df)
fair = fairness.run_all(res["X_test"], res["y_test"], res["score"])
print(dashboard.build(res, surv, fair))
