"""Churn classification: stratified CV + hold-out, three model families,
class-imbalance handling, probability calibration, 0-100 risk score,
risk tiers, and retention cost/benefit."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, precision_score,
                             recall_score, roc_auc_score, f1_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from . import config as C


def split(df: pd.DataFrame, test_size: float = 0.2, seed: int = C.SEED):
    X = df[C.FEATURES]
    y = df[C.LABEL].astype(int)
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def _preprocessor(scale: bool) -> ColumnTransformer:
    num_cols = C.NUMERIC_FEATURES + C.ENGINEERED_FEATURES
    num_tf = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        [("num", num_tf, num_cols),
         ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), C.CATEGORICAL_FEATURES)],
        verbose_feature_names_out=False,
    )


def make_models(y_train: pd.Series, seed: int = C.SEED) -> dict[str, Pipeline]:
    """Three model families; each handles the ~20% positive rate explicitly."""
    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    return {
        "logistic_regression": Pipeline([
            ("prep", _preprocessor(scale=True)),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0, random_state=seed)),
        ]),
        "random_forest": Pipeline([
            ("prep", _preprocessor(scale=False)),
            ("clf", RandomForestClassifier(n_estimators=500, min_samples_leaf=5, max_features="sqrt",
                                           class_weight="balanced_subsample", n_jobs=-1, random_state=seed)),
        ]),
        "xgboost": Pipeline([
            ("prep", _preprocessor(scale=False)),
            ("clf", XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=4, subsample=0.8,
                                  colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
                                  scale_pos_weight=pos_weight, eval_metric="logloss",
                                  n_jobs=-1, random_state=seed)),
        ]),
    }


SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
}


def cross_validate_models(models: dict, X_train, y_train, n_splits: int = 5, seed: int = C.SEED) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    for name, pipe in models.items():
        res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=1)
        row = {"model": name}
        for m in SCORING:
            row[f"{m}_mean"] = res[f"test_{m}"].mean()
            row[f"{m}_std"] = res[f"test_{m}"].std()
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def evaluate(y_true, proba, threshold: float = 0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred)),
        "brier": float(brier_score_loss(y_true, proba)),
        "threshold": float(threshold),
    }


def holdout_results(models: dict, X_train, y_train, X_test, y_test) -> tuple[pd.DataFrame, dict]:
    """Fit every model on the full training split, score the hold-out."""
    rows, probas = [], {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        p = pipe.predict_proba(X_test)[:, 1]
        probas[name] = p
        rows.append({"model": name, **evaluate(y_test, p)})
    return pd.DataFrame(rows).set_index("model"), probas


def calibrate(best_pipe: Pipeline, X_train, y_train, seed: int = C.SEED) -> CalibratedClassifierCV:
    """Isotonic calibration with inner CV: weighted models output inflated
    probabilities, so we re-map them before turning them into a risk score."""
    cal = CalibratedClassifierCV(best_pipe, method="isotonic",
                                 cv=StratifiedKFold(5, shuffle=True, random_state=seed))
    cal.fit(X_train, y_train)
    return cal


def calibration_table(y_true, proba, n_bins: int = 10) -> pd.DataFrame:
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"mean_predicted": mean_pred, "observed_rate": frac_pos})


def risk_score(proba: np.ndarray) -> np.ndarray:
    return np.clip(np.round(proba * 100), 0, 100).astype(int)


def risk_tier(score: np.ndarray) -> pd.Series:
    tiers = pd.Series(np.empty(len(score), dtype=object), name="RiskTier")
    for name, (lo, hi) in C.RISK_TIERS.items():
        tiers[(score >= lo) & (score < hi)] = name
    return pd.Categorical(tiers, categories=list(C.RISK_TIERS), ordered=True)


def tier_summary(y_true, score) -> pd.DataFrame:
    df = pd.DataFrame({"tier": risk_tier(score), "y": np.asarray(y_true)})
    base = df["y"].mean()
    g = df.groupby("tier", observed=False)["y"].agg(customers="size", churners="sum", churn_rate="mean")
    g["share_of_customers"] = g["customers"] / len(df)
    g["share_of_churners"] = g["churners"] / df["y"].sum()
    g["lift_vs_base"] = g["churn_rate"] / base
    return g.reset_index()


def retention_economics(y_true, score, offer_cost=C.RETENTION_OFFER_COST,
                        value=C.RETAINED_CUSTOMER_VALUE, success=C.OFFER_SUCCESS_RATE) -> pd.DataFrame:
    """Simple per-tier business case: offer every customer in the tier a
    retention incentive. Cost = customers * offer_cost. Benefit = true churners
    * success_rate * retained_value. Numbers are per hold-out set, scalable."""
    t = tier_summary(y_true, score)
    t["offer_cost"] = t["customers"] * offer_cost
    t["expected_saved_value"] = t["churners"] * success * value
    t["net_benefit"] = t["expected_saved_value"] - t["offer_cost"]
    t["roi"] = np.where(t["offer_cost"] > 0, t["net_benefit"] / t["offer_cost"], np.nan)
    return t


def best_cutoff_by_net_benefit(y_true, score, offer_cost=C.RETENTION_OFFER_COST,
                               value=C.RETAINED_CUSTOMER_VALUE, success=C.OFFER_SUCCESS_RATE) -> pd.DataFrame:
    """Sweep the score cutoff: target everyone with score >= cutoff."""
    y = np.asarray(y_true)
    rows = []
    for cut in range(0, 101, 5):
        mask = score >= cut
        n = int(mask.sum()); churners = int(y[mask].sum())
        rows.append({"cutoff": cut, "targeted": n, "churners_targeted": churners,
                     "recall": churners / y.sum() if y.sum() else 0.0,
                     "net_benefit": churners * success * value - n * offer_cost})
    return pd.DataFrame(rows)


def feature_importance(pipe: Pipeline) -> pd.DataFrame:
    names = pipe.named_steps["prep"].get_feature_names_out()
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
    else:
        imp = np.abs(clf.coef_[0])
    return (pd.DataFrame({"feature": names, "importance": imp})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def roc_points(y_true, proba) -> pd.DataFrame:
    fpr, tpr, _ = roc_curve(y_true, proba)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr})


def run_all(df: pd.DataFrame, seed: int = C.SEED) -> dict:
    """End-to-end modeling run. Returns a dict of tables/arrays used by the
    notebook, dashboard and README."""
    X_train, X_test, y_train, y_test = split(df, seed=seed)
    models = make_models(y_train, seed)
    cv_table = cross_validate_models(models, X_train, y_train, seed=seed)
    holdout, probas = holdout_results(models, X_train, y_train, X_test, y_test)
    best_name = holdout["roc_auc"].idxmax()
    cal = calibrate(make_models(y_train, seed)[best_name], X_train, y_train, seed)
    p_cal = cal.predict_proba(X_test)[:, 1]
    score = risk_score(p_cal)
    out = {
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "models": models, "cv_table": cv_table, "holdout": holdout, "probas": probas,
        "best_name": best_name, "calibrated_model": cal,
        "uncalibrated_eval": evaluate(y_test, probas[best_name]),
        "calibrated_eval": evaluate(y_test, p_cal),
        "calibration_before": calibration_table(y_test, probas[best_name]),
        "calibration_after": calibration_table(y_test, p_cal),
        "p_cal": p_cal, "score": score,
        "tiers": tier_summary(y_test, score),
        "economics": retention_economics(y_test, score),
        "cutoff_sweep": best_cutoff_by_net_benefit(y_test, score),
        "importance": feature_importance(models[best_name]),
        "roc": {n: roc_points(y_test, p) for n, p in probas.items()},
    }
    out["test_predictions"] = X_test.assign(Exited=y_test.values, churn_probability=p_cal,
                                            risk_score=score, risk_tier=risk_tier(score))
    return out


def save_outputs(res: dict, reports_dir=C.REPORTS) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    res["cv_table"].round(4).to_csv(reports_dir / "cv_results.csv")
    res["holdout"].round(4).to_csv(reports_dir / "holdout_results.csv")
    res["tiers"].round(4).to_csv(reports_dir / "risk_tiers.csv", index=False)
    res["economics"].round(2).to_csv(reports_dir / "retention_economics.csv", index=False)
    res["cutoff_sweep"].round(4).to_csv(reports_dir / "cutoff_sweep.csv", index=False)
    res["importance"].round(4).to_csv(reports_dir / "feature_importance.csv", index=False)
    res["test_predictions"].to_csv(reports_dir / "test_predictions.csv", index=False)
    metrics = {
        "best_model": res["best_name"],
        "cv": res["cv_table"].round(4).to_dict(orient="index"),
        "holdout": res["holdout"].round(4).to_dict(orient="index"),
        "best_uncalibrated": res["uncalibrated_eval"],
        "best_calibrated": res["calibrated_eval"],
        "tiers": res["tiers"].round(4).to_dict(orient="records"),
        "economics": res["economics"].round(2).to_dict(orient="records"),
        "best_cutoff": res["cutoff_sweep"].loc[res["cutoff_sweep"]["net_benefit"].idxmax()].to_dict(),
        "assumptions": {"offer_cost": C.RETENTION_OFFER_COST, "retained_value": C.RETAINED_CUSTOMER_VALUE,
                        "offer_success_rate": C.OFFER_SUCCESS_RATE},
        "n_train": int(len(res["y_train"])), "n_test": int(len(res["y_test"])),
    }
    with open(reports_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)
