import json
import os
import sys
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))
from data_loader import normalize_country  # noqa: E402

data_dir = os.path.join(ROOT_DIR, "app", "backend", "data")
os.makedirs(data_dir, exist_ok=True)

df = pd.read_csv(os.path.join(ROOT_DIR, "shipdata.csv"))

countries_clean = sorted({
    normalize_country(c) for c in df["country"].dropna() if str(c).strip()
})
suppliers_clean = sorted({
    str(s).strip().upper() for s in df["supplier"].dropna() if str(s).strip()
})
sap_codes_clean = sorted({
    str(x).strip().removesuffix(".0")
    for x in df["sap-code"].dropna()
    if str(x).strip()
})

with open(os.path.join(data_dir, "countries.json"), "w", encoding="utf-8") as f:
    json.dump(countries_clean, f, ensure_ascii=False, indent=2)

with open(os.path.join(data_dir, "suppliers.json"), "w", encoding="utf-8") as f:
    json.dump(suppliers_clean, f, ensure_ascii=False, indent=2)

with open(os.path.join(data_dir, "sap_codes.json"), "w", encoding="utf-8") as f:
    json.dump(sap_codes_clean, f, ensure_ascii=False, indent=2)

print(
    f"Successfully generated {len(countries_clean)} countries, "
    f"{len(suppliers_clean)} suppliers, and {len(sap_codes_clean)} SAP codes in {data_dir}"
)