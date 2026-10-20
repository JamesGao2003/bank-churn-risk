"""Project-wide constants: paths, label, feature lists, random seed."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT / "data" / "raw" / "Churn_Modelling.csv"
PROCESSED_PARQUET = ROOT / "data" / "processed" / "customers_clean.parquet"
DUCKDB_PATH = ROOT / "data" / "processed" / "churn.duckdb"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

SEED = 42
LABEL = "Exited"
TIME_COL = "Tenure"

# Identifier / free-text columns that must never reach a model (no predictive
# meaning, and Surname could act as a proxy for ethnicity/geography).
ID_COLS = ["RowNumber", "CustomerId", "Surname"]

NUMERIC_FEATURES = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
    "HasCrCard", "IsActiveMember", "EstimatedSalary",
]
CATEGORICAL_FEATURES = ["Geography", "Gender"]
ENGINEERED_FEATURES = ["BalanceSalaryRatio", "ZeroBalance", "ProductsPerTenureYear"]
FEATURES = NUMERIC_FEATURES + ENGINEERED_FEATURES + CATEGORICAL_FEATURES

# Risk score tiers on a 0-100 scale (score = 100 * calibrated P(churn)).
RISK_TIERS = {"Low": (0, 20), "Medium": (20, 50), "High": (50, 101)}

# Retention economics (assumptions, documented in README). Per customer.
RETENTION_OFFER_COST = 50.0      # cost of a retention offer (e.g. fee waiver)
RETAINED_CUSTOMER_VALUE = 500.0  # 12-month gross margin kept if the customer stays
OFFER_SUCCESS_RATE = 0.30        # share of true churners the offer actually retains

DATA_DICTIONARY = {
    "RowNumber": "Row index in the source file (identifier, dropped)",
    "CustomerId": "Unique customer identifier (dropped)",
    "Surname": "Customer surname (dropped: free text, potential ethnicity proxy)",
    "CreditScore": "Credit bureau score, 350-850",
    "Geography": "Country of the account: France, Germany, Spain",
    "Gender": "Female / Male (used only for fairness checks and as a model input, see model card)",
    "Age": "Customer age in years",
    "Tenure": "Years as a customer, 0-10 (survival-analysis time axis)",
    "Balance": "Account balance at snapshot",
    "NumOfProducts": "Number of bank products held, 1-4",
    "HasCrCard": "1 if the customer holds a credit card",
    "IsActiveMember": "1 if the customer is an active member",
    "EstimatedSalary": "Estimated annual salary",
    "Exited": "TARGET: 1 if the customer left the bank (churned), 0 otherwise",
}
