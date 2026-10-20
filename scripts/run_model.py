import sys, time; sys.path.insert(0, "src")
import pandas as pd
from churn import data, modeling
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
t=time.time(); df = data.build(); res = modeling.run_all(df); modeling.save_outputs(res)
print("elapsed", round(time.time()-t,1))
print(res["cv_table"].round(3)); print(res["holdout"].round(3)); print("best", res["best_name"])
print("uncal", res["uncalibrated_eval"]); print("cal", res["calibrated_eval"])
print(res["calibration_after"].round(3)); print(res["tiers"].round(3)); print(res["economics"].round(1)); print(res["cutoff_sweep"].round(2)); print(res["importance"].head(10))
