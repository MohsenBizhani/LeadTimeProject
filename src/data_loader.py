"""
Data loading and preprocessing module for lead time prediction.
Handles raw CSV loading, text canonicalization, datetime conversions,
and validation of milestone consistency.
"""

from pathlib import Path
import pandas as pd


COUNTRY_MAP = {
    "UNITED ARAB EMIRATES": "U.A.E",
    "UAE": "U.A.E",
    "U.A.E.": "U.A.E",
}


def normalize_country(val: str) -> str:
    cleaned = str(val).strip().upper()
    return COUNTRY_MAP.get(cleaned, cleaned)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans an in-memory raw dataframe with the same rules as load_clean_data.
    Required raw columns: sap-code, country, supplier, weight, registery-date,
    ministry-date, ship-date, costums-arrive-date, factory-arrive-date, LeadTime.
    """
    required = [
        "sap-code", "country", "supplier", "weight", "registery-date",
        "ministry-date", "ship-date", "costums-arrive-date",
        "factory-arrive-date", "LeadTime",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()

    # Drop rows missing critical identifiers or target
    df = df.dropna(subset=required).copy()
    if df.empty:
        raise ValueError("Dataset has no complete rows after removing missing values.")

    # 1. Clean categorical text
    df["country_clean"] = df["country"].map(normalize_country)
    df["supplier_clean"] = df["supplier"].astype(str).str.strip().str.upper()
    df["sap_code_clean"] = (
        df["sap-code"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df["LeadTime"] = pd.to_numeric(df["LeadTime"], errors="coerce")
    if df["weight"].isna().any() or df["LeadTime"].isna().any():
        raise ValueError("Invalid numeric values found in weight or LeadTime.")
    if (df["weight"] <= 0).any():
        raise ValueError("All rows must have positive weight.")

    # 2. Parse dates
    date_cols = [
        "registery-date",
        "ministry-date",
        "ship-date",
        "costums-arrive-date",
        "factory-arrive-date",
    ]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], format="mixed", errors="coerce")
        if df[col].isna().any():
            raise ValueError(f"Invalid or unparseable dates found in column: {col}")

    # 3. Compute milestone segment durations
    df["dur_reg_to_min"] = (df["ministry-date"] - df["registery-date"]).dt.days
    df["dur_min_to_ship"] = (df["ship-date"] - df["ministry-date"]).dt.days
    df["dur_ship_to_cust"] = (df["costums-arrive-date"] - df["ship-date"]).dt.days
    df["dur_cust_to_fact"] = (df["factory-arrive-date"] - df["costums-arrive-date"]).dt.days

    for dur_col in ("dur_reg_to_min", "dur_min_to_ship", "dur_ship_to_cust", "dur_cust_to_fact"):
        if (df[dur_col] < 0).any():
            raise ValueError(f"Negative milestone duration detected in {dur_col}.")

    # Cumulative elapsed days from registry
    df["elapsed_reg_to_ship"] = (df["ship-date"] - df["registery-date"]).dt.days
    df["elapsed_reg_to_cust"] = (df["costums-arrive-date"] - df["registery-date"]).dt.days

    # 4. Integrity assertions
    computed_lead = (df["factory-arrive-date"] - df["registery-date"]).dt.days
    if not (computed_lead == df["LeadTime"].round().astype(int)).all():
        raise ValueError("Integrity check failed: LeadTime mismatch.")
    df["LeadTime"] = df["LeadTime"].astype(float)

    return df.reset_index(drop=True)


def load_clean_data(filepath: str = "shipdata.csv") -> pd.DataFrame:
    """Loads and preprocesses shipdata.csv."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")

    df = pd.read_csv(path)
    return clean_dataframe(df)


if __name__ == "__main__":
    df = load_clean_data()
    print(f"Data successfully loaded. Shape: {df.shape}")
    print(f"Unique countries: {df['country_clean'].unique().tolist()}")
    print(f"Unique suppliers: {df['supplier_clean'].nunique()}")
    print(f"Unique SAP codes: {df['sap_code_clean'].nunique()}")