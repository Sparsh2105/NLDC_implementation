"""
Build SRD and GF composite indices.

IMPORTANT: Replace the raw column names below with columns in your real dataset.
The paper's Table 1 lists DPSIR SRD indicators; Table 2 lists six GF dimensions.
No raw observations are supplied in this repository.
"""
import numpy as np
import pandas as pd
from common import load_panel, OUT, require_columns

# Map each raw SRD indicator to polarity: +1 = positive, -1 = negative.
# Names here are illustrative placeholders. Edit them to match your source data.
SRD_INDICATORS = {
    # Driving
    "agri_output_per_rural_person": +1,
    "rural_disposable_income": +1,
    "agriculture_share_primary_output": +1,
    "rural_self_employment_rate": +1,
    "rural_schooling_years": +1,
    "rural_poverty_incidence": -1,
    # Pressure
    "plastic_film_per_area": -1,
    "pesticide_use_per_area": -1,
    "fertilizer_use_per_area": -1,
    "crop_disaster_affected_area": -1,
    "rural_water_consumption": -1,
    "rural_electricity_consumption": -1,
    # State
    "food_production": +1,
    "agri_mechanization_power": +1,
    "effective_irrigated_area": +1,
    "arable_land_productivity": +1,
    "agri_labor_productivity": +1,
    # Impact
    "agriculture_industry_ratio": +1,
    "agri_industry_coordination": +1,
    "urban_rural_consumption_ratio": -1,
    "rural_housing_area": +1,
    "rural_engel_coefficient": -1,
    # Response
    "rural_education_expenditure_share": +1,
    "agri_fiscal_expenditure_share": +1,
    "village_environment_improvement_rate": +1,
    "forest_coverage_rate": +1,
    "rural_sewage_treatment_rate": +1,
    "rural_waste_disposal_rate": +1,
    "rural_health_technicians_per_1000": +1,
    "rural_sanitation_rate": +1,
}

# Six green finance dimensions from the paper's Table 2.
# These are raw columns or pre-calculated component scores; update as needed.
GF_COMPONENTS = [
    "green_credit",
    "green_investment",
    "green_insurance",
    "green_support",
    "green_fund",
    "green_equity",
]

def minmax_polarity(series, polarity):
    x = pd.to_numeric(series, errors="coerce")
    lo, hi = x.min(skipna=True), x.max(skipna=True)
    if pd.isna(lo) or pd.isna(hi):
        return pd.Series(np.nan, index=x.index)
    if np.isclose(hi, lo):
        return pd.Series(0.0, index=x.index)
    if polarity > 0:
        return (x - lo) / (hi - lo)
    return (hi - x) / (hi - lo)

def entropy_weights(norm_df):
    """Standard entropy weighting; columns must be nonnegative."""
    z = norm_df.astype(float).clip(lower=0)
    eps = 1e-12
    p = z.div(z.sum(axis=0).replace(0, np.nan), axis=1).fillna(0)
    n = len(z)
    if n <= 1:
        raise ValueError("Entropy weighting requires at least two observations.")
    k = 1.0 / np.log(n)
    entropy = -k * (p * np.log(p + eps)).sum(axis=0)
    diversity = (1 - entropy).clip(lower=0)
    if diversity.sum() == 0:
        return pd.Series(1 / len(diversity), index=diversity.index)
    return diversity / diversity.sum()

def weighted_index(norm_df, weights):
    return norm_df.mul(weights, axis=1).sum(axis=1, min_count=1)

def main():
    df = load_panel()
    srd_cols = list(SRD_INDICATORS)
    gf_cols = GF_COMPONENTS
    require_columns(df, srd_cols + gf_cols)

    srd_norm = pd.DataFrame(index=df.index)
    for col, polarity in SRD_INDICATORS.items():
        srd_norm[col] = minmax_polarity(df[col], polarity)

    srd_w = entropy_weights(srd_norm)
    df["SRD"] = weighted_index(srd_norm, srd_w)
    srd_w.rename("weight").to_csv(OUT / "srd_entropy_weights.csv")

    gf_norm = pd.DataFrame(index=df.index)
    for col in gf_cols:
        gf_norm[col] = minmax_polarity(df[col], +1)
    gf_w = entropy_weights(gf_norm)
    df["GF"] = weighted_index(gf_norm, gf_w)
    gf_w.rename("weight").to_csv(OUT / "gf_entropy_weights.csv")

    df.to_csv(OUT / "panel_with_indices.csv", index=False)
    print("Saved outputs/panel_with_indices.csv and index weight tables.")
    print("Check the paper's precise weighting formula and your source data definitions before reporting results.")

if __name__ == "__main__":
    main()
