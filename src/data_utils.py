"""Loading and basic preparation of the CFPB complaint sample."""
import pandas as pd

TEXT_COL = "Consumer complaint narrative"
TARGET_COL = "Product"

# Short names so the 11 product labels fit on chart axes
SHORT_NAMES = {
    "Credit reporting or other personal consumer reports": "Credit reporting",
    "Debt collection": "Debt collection",
    "Credit card": "Credit card",
    "Checking or savings account": "Checking/savings",
    "Money transfer, virtual currency, or money service": "Money transfer",
    "Mortgage": "Mortgage",
    "Vehicle loan or lease": "Vehicle loan",
    "Student loan": "Student loan",
    "Payday loan, title loan, personal loan, or advance loan": "Payday/personal loan",
    "Prepaid card": "Prepaid card",
    "Debt or credit management": "Debt management",
}


def load_complaints(path):
    """Read the sample CSV and add a few helper columns used throughout the notebook."""
    df = pd.read_csv(path, dtype=str)
    df["Date received"] = pd.to_datetime(df["Date received"])
    df["month"] = df["Date received"].dt.to_period("M")
    df["product_short"] = df[TARGET_COL].map(SHORT_NAMES)
    df[TEXT_COL] = df[TEXT_COL].fillna("").str.strip()
    df["word_count"] = df[TEXT_COL].str.split().str.len()
    return df


def missing_values(df):
    """Table of columns that have missing values: count and share of rows."""
    counts = df.isna().sum()
    counts = counts[counts > 0].sort_values(ascending=False)
    return pd.DataFrame({"missing_rows": counts,
                         "share_missing": (counts / len(df)).round(3)})


def duplicate_narratives(df):
    """Narratives that appear more than once (copy-pasted template letters)."""
    counts = df[TEXT_COL].value_counts()
    repeated = counts[counts > 1]
    return pd.DataFrame({"copies": repeated.values,
                         "narrative_start": [t[:120] + "..." for t in repeated.index]})
