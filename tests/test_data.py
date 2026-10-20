import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from churn import config as C  # noqa: E402
from churn import data  # noqa: E402


@pytest.fixture(scope="module")
def raw():
    return data.load_raw()


@pytest.fixture(scope="module")
def clean(raw):
    return data.clean(raw)


def test_row_count(raw, clean):
    assert len(raw) == 10_000
    assert len(clean) == 10_000, "cleaning must not drop rows"


def test_schema(raw):
    assert list(raw.columns) == list(C.DATA_DICTIONARY.keys())


def test_no_missing(clean):
    assert clean.isna().sum().sum() == 0


def test_label_distribution(clean):
    rate = clean[C.LABEL].mean()
    assert set(clean[C.LABEL].unique()) == {0, 1}
    assert 0.15 < rate < 0.30, f"churn rate {rate:.3f} outside expected band"
    assert clean[C.LABEL].sum() == 2037


def test_no_leakage_or_id_fields_in_features():
    leaked = set(C.FEATURES) & (set(C.ID_COLS) | {C.LABEL})
    assert not leaked, f"identifier/label columns in feature list: {leaked}"


def test_features_exist_and_are_not_label(clean):
    for f in C.FEATURES:
        assert f in clean.columns, f
        # no numeric feature may be a copy of the label
        if f in C.NUMERIC_FEATURES + C.ENGINEERED_FEATURES:
            assert not np.array_equal(clean[f].to_numpy(dtype=float), clean[C.LABEL].to_numpy(dtype=float))


def test_engineered_features(clean):
    assert (clean["ZeroBalance"] == (clean["Balance"] == 0).astype(int)).all()
    assert (clean["BalanceSalaryRatio"] >= 0).all()
    assert (clean["ProductsPerTenureYear"] > 0).all()


def test_value_ranges(clean):
    assert clean["CreditScore"].between(300, 850).all()
    assert clean["Age"].between(18, 100).all()
    assert clean["Tenure"].between(0, 10).all()
    assert set(clean["Geography"].cat.categories) == {"France", "Germany", "Spain"}
    assert set(clean["Gender"].cat.categories) == {"Female", "Male"}


def test_duckdb_roundtrip(tmp_path):
    pq = tmp_path / "c.parquet"
    db = tmp_path / "c.duckdb"
    df = data.build(parquet_path=pq, duckdb_path=db)
    n = data.sql("SELECT COUNT(*) AS n FROM customers", duckdb_path=db)["n"].iloc[0]
    assert n == len(df) == 10_000
    rate = data.sql(f"SELECT AVG({C.LABEL}) AS r FROM customers", duckdb_path=db)["r"].iloc[0]
    assert abs(rate - df[C.LABEL].mean()) < 1e-9
