import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from churn import config as C  # noqa: E402
from churn import data, modeling, survival, fairness  # noqa: E402


@pytest.fixture(scope="module")
def df():
    return data.clean(data.load_raw())


@pytest.fixture(scope="module")
def split(df):
    return modeling.split(df)


def test_split_is_stratified(split):
    X_train, X_test, y_train, y_test = split
    assert len(X_train) == 8000 and len(X_test) == 2000
    assert abs(y_train.mean() - y_test.mean()) < 0.01
    assert C.LABEL not in X_train.columns, "label leaked into features"
    assert not set(C.ID_COLS) & set(X_train.columns), "identifier leaked into features"


def test_models_beat_baseline_and_output_valid_probabilities(split):
    X_train, X_test, y_train, y_test = split
    models = modeling.make_models(y_train)
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        p = pipe.predict_proba(X_test)[:, 1]
        assert np.all((p >= 0) & (p <= 1))
        m = modeling.evaluate(y_test, p)
        assert m["roc_auc"] > 0.70, f"{name} AUC {m['roc_auc']:.3f} too low"
        assert m["recall"] > 0.40, f"{name} recall {m['recall']:.3f}: imbalance handling failed"


def test_risk_score_and_tiers():
    p = np.array([0.0, 0.19, 0.2, 0.494, 0.5, 1.0])
    s = modeling.risk_score(p)
    assert s.min() >= 0 and s.max() <= 100
    tiers = list(modeling.risk_tier(s))
    assert tiers == ["Low", "Low", "Medium", "Medium", "High", "High"]


def test_economics_arithmetic():
    y = np.array([1, 0, 1, 0, 0, 1])
    score = np.array([90, 90, 10, 10, 10, 10])
    t = modeling.retention_economics(y, score, offer_cost=10, value=100, success=0.5)
    high = t.set_index("tier").loc["High"]
    assert high["customers"] == 2 and high["churners"] == 1
    assert high["offer_cost"] == 20 and high["expected_saved_value"] == 50 and high["net_benefit"] == 30


def test_hazard_table_consistency(df):
    hz = survival.hazard_by_tenure(df)
    assert hz["observed"].sum() == df[C.LABEL].sum()
    assert hz["at_risk"].iloc[0] == len(df)
    assert hz["hazard"].dropna().between(0, 1).all()


def test_cox_fits_and_reports_hr(df):
    cph = survival.fit_cox(df.sample(3000, random_state=0))
    t = survival.cox_table(cph)
    assert set(t["covariate"]) == set(survival.COX_COVARIATES)
    assert (t["hazard_ratio"] > 0).all() and t["p_value"].between(0, 1).all()


def test_fairness_metrics_bounds():
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    score = np.array([80, 80, 10, 10, 80, 10, 80, 10])
    g = np.array(["a", "a", "a", "a", "b", "b", "b", "b"])
    t = fairness.group_metrics(y, score, g)
    assert len(t) == 2
    for col in ["recall", "false_positive_rate", "precision", "selection_rate"]:
        assert t[col].between(0, 1).all()
    assert t.set_index("group").loc["b", "recall"] == 1.0
    assert t.set_index("group").loc["a", "false_positive_rate"] == 0.5
