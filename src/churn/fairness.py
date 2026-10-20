"""Fairness check: compare recall (true positive rate), false positive rate,
precision and selection rate across Gender and age bands on the hold-out set."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .data import age_band

HIGH_TIER_CUTOFF = C.RISK_TIERS["High"][0]


def group_metrics(y_true, score, groups, cutoff: int = HIGH_TIER_CUTOFF) -> pd.DataFrame:
    y = np.asarray(y_true).astype(int)
    pred = (np.asarray(score) >= cutoff).astype(int)
    df = pd.DataFrame({"y": y, "pred": pred, "group": np.asarray(groups)})
    rows = []
    for g, d in df.groupby("group", observed=True):
        tp = int(((d.y == 1) & (d.pred == 1)).sum()); fn = int(((d.y == 1) & (d.pred == 0)).sum())
        fp = int(((d.y == 0) & (d.pred == 1)).sum()); tn = int(((d.y == 0) & (d.pred == 0)).sum())
        rows.append({
            "group": str(g), "n": len(d), "actual_churn_rate": d.y.mean(),
            "selection_rate": d.pred.mean(),
            "recall": tp / (tp + fn) if tp + fn else np.nan,
            "false_positive_rate": fp / (fp + tn) if fp + tn else np.nan,
            "precision": tp / (tp + fp) if tp + fp else np.nan,
        })
    return pd.DataFrame(rows)


def gaps(table: pd.DataFrame) -> dict:
    return {
        "recall_gap": float(table["recall"].max() - table["recall"].min()),
        "fpr_gap": float(table["false_positive_rate"].max() - table["false_positive_rate"].min()),
        "selection_rate_ratio": float(table["selection_rate"].min() / table["selection_rate"].max())
        if table["selection_rate"].max() > 0 else np.nan,
    }


def run_all(X_test: pd.DataFrame, y_test, score, cutoff: int = HIGH_TIER_CUTOFF) -> dict:
    by_gender = group_metrics(y_test, score, X_test["Gender"].astype(str).values, cutoff)
    by_age = group_metrics(y_test, score, age_band(X_test["Age"]).astype(str).values, cutoff)
    by_age["order"] = by_age["group"].map({"18-30": 0, "31-40": 1, "41-50": 2, "51-60": 3, "60+": 4})
    by_age = by_age.sort_values("order").drop(columns="order").reset_index(drop=True)
    return {"cutoff": cutoff, "by_gender": by_gender, "by_age": by_age,
            "gender_gaps": gaps(by_gender), "age_gaps": gaps(by_age)}


def save_outputs(res: dict, reports_dir=C.REPORTS) -> None:
    res["by_gender"].round(4).to_csv(reports_dir / "fairness_by_gender.csv", index=False)
    res["by_age"].round(4).to_csv(reports_dir / "fairness_by_age.csv", index=False)
