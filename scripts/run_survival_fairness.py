import sys; sys.path.insert(0, "src")
import pandas as pd
from churn import data, modeling, survival, fairness
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
df = data.load_clean()
s = survival.run_all(df); survival.save_outputs(s)
print("logrank", s["logrank_p"]); print(s["hazard"].round(4)); print("peak", s["peak_hazard_tenure"], s["peak_hazard"])
print(s["cox_table"].round(4)); print("concordance", round(s["concordance"],4)); print(s["ph_test"].round(4)); print(s["median_survival"])
res = modeling.run_all(df)
f = fairness.run_all(res["X_test"], res["y_test"], res["score"]); fairness.save_outputs(f)
print(f["by_gender"].round(3)); print(f["gender_gaps"]); print(f["by_age"].round(3)); print(f["age_gaps"])
