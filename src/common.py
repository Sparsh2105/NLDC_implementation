from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "panel_raw.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

CONTROLS = ["URB", "UED", "HUC", "FDL", "OPE"]
ID_COLS = ["city_id", "city_name", "province", "year"]

def load_panel(path=DATA):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Missing {path}. Copy your real city-year data to data/panel_raw.csv. "
            "The repository intentionally contains no fabricated research data."
        )
    df = pd.read_csv(path)
    required = {"city_id", "province", "year"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df["year"] = pd.to_numeric(df["year"], errors="raise").astype(int)
    return df.sort_values(["city_id", "year"]).reset_index(drop=True)

def require_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def significance(p):
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""
