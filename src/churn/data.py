"""Data pipeline: load raw CSV, validate schema, clean, engineer features,
persist to Parquet, and register the table in DuckDB for SQL access."""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from . import config as C

EXPECTED_COLUMNS = list(C.DATA_DICTIONARY.keys())

CLEANING_RULES = [
    "Drop identifier columns RowNumber, CustomerId, Surname (no signal, leakage/proxy risk).",
    "Assert no missing values; assert CustomerId unique; assert Exited in {0,1}.",
    "Clip CreditScore to [300, 850]; Age to [18, 100] (no rows affected in this dataset).",
    "Cast Geography/Gender to category; HasCrCard/IsActiveMember/Exited to int8.",
    "Engineer BalanceSalaryRatio = Balance / EstimatedSalary, ZeroBalance = 1[Balance == 0], "
    "ProductsPerTenureYear = NumOfProducts / (Tenure + 1).",
]


def load_raw(path=C.RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Raw file is missing columns: {sorted(missing)}")
    return df


def validate_raw(df: pd.DataFrame) -> None:
    assert df.isna().sum().sum() == 0, "unexpected missing values"
    assert df["CustomerId"].is_unique, "CustomerId must be unique"
    assert set(df[C.LABEL].unique()) <= {0, 1}, "label must be binary"
    assert df["Tenure"].between(0, 10).all(), "Tenure outside 0-10"
    assert df["NumOfProducts"].between(1, 4).all(), "NumOfProducts outside 1-4"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    validate_raw(df)
    out = df.drop(columns=C.ID_COLS).copy()
    out["CreditScore"] = out["CreditScore"].clip(300, 850)
    out["Age"] = out["Age"].clip(18, 100)
    for col in ["Geography", "Gender"]:
        out[col] = out[col].astype("category")
    for col in ["HasCrCard", "IsActiveMember", C.LABEL]:
        out[col] = out[col].astype("int8")
    out["BalanceSalaryRatio"] = out["Balance"] / out["EstimatedSalary"].clip(lower=1.0)
    out["ZeroBalance"] = (out["Balance"] == 0).astype("int8")
    out["ProductsPerTenureYear"] = out["NumOfProducts"] / (out["Tenure"] + 1)
    return out


def build(raw_path=C.RAW_CSV, parquet_path=C.PROCESSED_PARQUET, duckdb_path=C.DUCKDB_PATH) -> pd.DataFrame:
    """Full pipeline. Writes Parquet + DuckDB and returns the clean frame."""
    df = clean(load_raw(raw_path))
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    con = duckdb.connect(str(duckdb_path))
    con.execute("CREATE OR REPLACE TABLE customers AS SELECT * FROM read_parquet(?)", [str(parquet_path)])
    con.close()
    return df


def sql(query: str, duckdb_path=C.DUCKDB_PATH) -> pd.DataFrame:
    """Run a SQL query against the processed DuckDB table `customers`."""
    con = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        return con.execute(query).df()
    finally:
        con.close()


def load_clean(parquet_path=C.PROCESSED_PARQUET) -> pd.DataFrame:
    if not parquet_path.exists():
        return build()
    return pd.read_parquet(parquet_path)


def age_band(age: pd.Series) -> pd.Series:
    bins = [17, 30, 40, 50, 60, 120]
    labels = ["18-30", "31-40", "41-50", "51-60", "60+"]
    return pd.cut(age, bins=bins, labels=labels)


def eda_summary(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Churn rate by key segments, computed in DuckDB SQL for reproducibility."""
    con = duckdb.connect()
    con.register("c", df)
    out = {}
    for col in ["Geography", "Gender", "NumOfProducts", "IsActiveMember", "Tenure"]:
        out[col] = con.execute(
            f"SELECT {col}, COUNT(*) AS n, ROUND(AVG({C.LABEL}), 4) AS churn_rate "
            f"FROM c GROUP BY {col} ORDER BY {col}"
        ).df()
    out["AgeBand"] = (
        df.assign(AgeBand=age_band(df["Age"]))
        .groupby("AgeBand", observed=True)[C.LABEL]
        .agg(n="size", churn_rate="mean")
        .round(4)
        .reset_index()
    )
    con.close()
    return out
