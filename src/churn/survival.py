"""Survival analysis: Kaplan-Meier curves (stratified), log-rank tests,
Cox proportional-hazards model, and the hazard-by-tenure profile.

Time axis = Tenure (years with the bank, 0-10); event = Exited (1 = churned).
Customers still with the bank are right-censored at their current tenure.
This is a cross-sectional snapshot read as survival data, which is a known
simplification (see README limitations)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test, proportional_hazard_test

from . import config as C

STRATA = ["Geography", "NumOfProducts", "IsActiveMember"]

COX_COVARIATES = ["Age", "CreditScore", "Balance_k", "EstimatedSalary_k", "HasCrCard",
                  "IsActiveMember", "Products_2", "Products_3plus", "Geography_Germany",
                  "Geography_Spain", "Gender_Male"]


def km_overall(df: pd.DataFrame) -> KaplanMeierFitter:
    kmf = KaplanMeierFitter(label="All customers")
    kmf.fit(df[C.TIME_COL], df[C.LABEL])
    return kmf


def km_by(df: pd.DataFrame, col: str) -> tuple[dict[str, KaplanMeierFitter], float]:
    """KM fit per level of `col`, plus the multivariate log-rank p-value."""
    fits = {}
    for level, g in df.groupby(col, observed=True):
        kmf = KaplanMeierFitter(label=f"{col}={level}")
        kmf.fit(g[C.TIME_COL], g[C.LABEL])
        fits[str(level)] = kmf
    p = multivariate_logrank_test(df[C.TIME_COL], df[col], df[C.LABEL]).p_value
    return fits, float(p)


def km_table(fits: dict[str, KaplanMeierFitter]) -> pd.DataFrame:
    """Long table of survival curves for plotting: tenure, survival, group."""
    rows = []
    for name, kmf in fits.items():
        sf = kmf.survival_function_.reset_index()
        sf.columns = ["tenure", "survival"]
        ci = kmf.confidence_interval_.reset_index(drop=True)
        sf["lower"] = ci.iloc[:, 0].values
        sf["upper"] = ci.iloc[:, 1].values
        sf["group"] = name
        rows.append(sf)
    return pd.concat(rows, ignore_index=True)


def hazard_by_tenure(df: pd.DataFrame) -> pd.DataFrame:
    """Discrete hazard per tenure year: churners at t / customers at risk at t."""
    kmf = km_overall(df)
    et = kmf.event_table.reset_index()
    et = et.rename(columns={"event_at": "tenure"})
    et["hazard"] = et["observed"] / et["at_risk"].replace(0, np.nan)
    return et[["tenure", "at_risk", "observed", "censored", "hazard"]]


def cox_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        C.TIME_COL: df[C.TIME_COL].astype(float),
        C.LABEL: df[C.LABEL].astype(int),
        "Age": df["Age"].astype(float),
        "CreditScore": df["CreditScore"].astype(float),
        "Balance_k": df["Balance"] / 1000.0,
        "EstimatedSalary_k": df["EstimatedSalary"] / 1000.0,
        "HasCrCard": df["HasCrCard"].astype(int),
        "IsActiveMember": df["IsActiveMember"].astype(int),
        "Products_2": (df["NumOfProducts"] == 2).astype(int),
        "Products_3plus": (df["NumOfProducts"] >= 3).astype(int),
        "Geography_Germany": (df["Geography"] == "Germany").astype(int),
        "Geography_Spain": (df["Geography"] == "Spain").astype(int),
        "Gender_Male": (df["Gender"] == "Male").astype(int),
    })
    return out


def fit_cox(df: pd.DataFrame, penalizer: float = 0.01) -> CoxPHFitter:
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(cox_frame(df), duration_col=C.TIME_COL, event_col=C.LABEL)
    return cph


def cox_table(cph: CoxPHFitter) -> pd.DataFrame:
    s = cph.summary
    t = pd.DataFrame({
        "covariate": s.index,
        "hazard_ratio": s["exp(coef)"].values,
        "hr_lower95": s["exp(coef) lower 95%"].values,
        "hr_upper95": s["exp(coef) upper 95%"].values,
        "p_value": s["p"].values,
    })
    t["significant_5pct"] = t["p_value"] < 0.05
    return t.sort_values("hazard_ratio", ascending=False).reset_index(drop=True)


def ph_assumption(cph: CoxPHFitter, df: pd.DataFrame) -> pd.DataFrame:
    res = proportional_hazard_test(cph, cox_frame(df), time_transform="rank")
    s = res.summary.reset_index().rename(columns={"index": "covariate", "p": "p_value"})
    return s[["covariate", "test_statistic", "p_value"]]


def run_all(df: pd.DataFrame) -> dict:
    out = {"km_overall": km_overall(df), "km": {}, "logrank_p": {}}
    for col in STRATA:
        fits, p = km_by(df, col)
        out["km"][col] = fits
        out["logrank_p"][col] = p
    out["hazard"] = hazard_by_tenure(df)
    hz = out["hazard"].dropna(subset=["hazard"])
    out["peak_hazard_tenure"] = int(hz.loc[hz["hazard"].idxmax(), "tenure"])
    out["peak_hazard"] = float(hz["hazard"].max())
    cph = fit_cox(df)
    out["cox"] = cph
    out["cox_table"] = cox_table(cph)
    out["concordance"] = float(cph.concordance_index_)
    out["ph_test"] = ph_assumption(cph, df)
    out["median_survival"] = {k: {g: float(f.median_survival_time_) for g, f in v.items()}
                              for k, v in out["km"].items()}
    return out


def save_outputs(res: dict, reports_dir=C.REPORTS) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    res["cox_table"].round(4).to_csv(reports_dir / "cox_hazard_ratios.csv", index=False)
    res["hazard"].round(4).to_csv(reports_dir / "hazard_by_tenure.csv", index=False)
    res["ph_test"].round(4).to_csv(reports_dir / "cox_ph_assumption_test.csv", index=False)
    for col, fits in res["km"].items():
        km_table(fits).round(4).to_csv(reports_dir / f"km_by_{col}.csv", index=False)
    pd.Series(res["logrank_p"], name="logrank_p").round(6).to_csv(reports_dir / "logrank_tests.csv")
