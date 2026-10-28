"""Baseline, moderation, robustness, DID, and IV models."""
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from linearmodels.panel import PanelOLS
from linearmodels.iv import IV2SLS
from common import load_panel, OUT, CONTROLS, require_columns

PILOT_PROVINCES = {"Zhejiang", "Jiangxi", "Guangdong", "Guizhou", "Xinjiang"}

def save_summary(result, name):
    with open(OUT / f"{name}_summary.txt", "w", encoding="utf-8") as f:
        f.write(str(result.summary))

def panel_formula(df, formula, name):
    required = ["city_id", "year"]
    d = df.dropna(subset=[c for c in ["SRD", "GF"] + CONTROLS if c in df.columns]).copy()
    d = d.set_index(["city_id", "year"]).sort_index()
    mod = PanelOLS.from_formula(formula, data=d, drop_absorbed=True)
    res = mod.fit(cov_type="clustered", cluster_entity=True)
    save_summary(res, name)
    return res

def coefficient_table(results):
    rows = []
    for model_name, res in results.items():
        for term in res.params.index:
            rows.append({
                "model": model_name,
                "term": term,
                "coef": res.params[term],
                "std_error": res.std_errors[term] if hasattr(res, "std_errors") else np.nan,
                "p_value": res.pvalues[term],
            })
    return pd.DataFrame(rows)

def main():
    df = load_panel()
    require_columns(df, ["SRD", "GF", "ER", "DE"] + CONTROLS)
    df["GF_ER"] = df["GF"] * df["ER"]
    df["GF_DE"] = df["GF"] * df["DE"]

    results = {}
    results["baseline_twfe"] = panel_formula(
        df, "SRD ~ 1 + GF + URB + UED + HUC + FDL + OPE + EntityEffects + TimeEffects",
        "baseline_twfe"
    )
    results["moderation_ER"] = panel_formula(
        df, "SRD ~ 1 + GF + ER + GF_ER + URB + UED + HUC + FDL + OPE + EntityEffects + TimeEffects",
        "moderation_ER"
    )
    results["moderation_DE"] = panel_formula(
        df, "SRD ~ 1 + GF + DE + GF_DE + URB + UED + HUC + FDL + OPE + EntityEffects + TimeEffects",
        "moderation_DE"
    )

    # Robustness 1: lag SRD by one year as dependent variable, matching the paper's described check.
    lag_df = df.sort_values(["city_id", "year"]).copy()
    lag_df["SRD_lag1"] = lag_df.groupby("city_id")["SRD"].shift(1)
    lag_df = lag_df.dropna(subset=["SRD_lag1"])
    lag_df = lag_df.set_index(["city_id", "year"]).sort_index()
    lag_mod = PanelOLS.from_formula(
        "SRD_lag1 ~ 1 + GF + URB + UED + HUC + FDL + OPE + EntityEffects + TimeEffects",
        data=lag_df, drop_absorbed=True
    )
    results["lagged_SRD"] = lag_mod.fit(cov_type="clustered", cluster_entity=True)
    save_summary(results["lagged_SRD"], "lagged_SRD")

    # Robustness 2: 3% bilateral winsorization of continuous analysis variables.
    wins = df.copy()
    for col in ["SRD", "GF"] + CONTROLS:
        lo, hi = wins[col].quantile([0.03, 0.97])
        wins[col] = wins[col].clip(lo, hi)
    wins_res = panel_formula(
        wins, "SRD ~ 1 + GF + URB + UED + HUC + FDL + OPE + EntityEffects + TimeEffects",
        "winsorized_3pct"
    )
    results["winsorized_3pct"] = wins_res

    # Robustness 3: PCA-based GF index. Components should be consistently scaled first.
    pca_cols = [c for c in ["green_credit", "green_investment", "green_insurance",
                            "green_support", "green_fund", "green_equity"] if c in df.columns]
    if len(pca_cols) >= 2:
        temp = df[pca_cols].apply(pd.to_numeric, errors="coerce")
        temp = temp.fillna(temp.median())
        z = StandardScaler().fit_transform(temp)
        df["GF_PCA"] = PCA(n_components=1).fit_transform(z)[:, 0]
        pca_df = df.copy().set_index(["city_id", "year"]).sort_index()
        pca_res = PanelOLS.from_formula(
            "SRD ~ 1 + GF_PCA + URB + UED + HUC + FDL + OPE + EntityEffects + TimeEffects",
            data=pca_df, drop_absorbed=True
        ).fit(cov_type="clustered", cluster_entity=True)
        save_summary(pca_res, "pca_GF_robustness")
        results["pca_GF_robustness"] = pca_res
    else:
        warnings.warn("Skipping PCA GF robustness: need at least two GF component columns.")

    # DID: treatment provinces × post-2017. Include province and year fixed effects.
    did = df.copy()
    did["treated"] = did["province"].isin(PILOT_PROVINCES).astype(int)
    did["post2017"] = (did["year"] >= 2017).astype(int)
    did["DID"] = did["treated"] * did["post2017"]
    # City FE absorbs treated; year FE absorbs post2017; DID remains identifiable.
    did = did.set_index(["city_id", "year"]).sort_index()
    did_res = PanelOLS.from_formula(
        "SRD ~ 1 + DID + GF + URB + UED + HUC + FDL + OPE + EntityEffects + TimeEffects",
        data=did, drop_absorbed=True
    ).fit(cov_type="clustered", cluster_entity=True)
    save_summary(did_res, "DID_2017_pilots")
    results["DID_2017_pilots"] = did_res

    # IV/2SLS: entity/year dummy variables included explicitly.
    # This is computationally heavier; instrument validity must be justified separately.
    if "bank_financial_institutions" in df.columns:
        iv = df.dropna(subset=["SRD", "GF", "bank_financial_institutions"] + CONTROLS).copy()
        iv["city_fe"] = iv["city_id"].astype(str)
        iv["year_fe"] = iv["year"].astype(str)
        exog = pd.concat([
            iv[CONTROLS].reset_index(drop=True),
            pd.get_dummies(iv["city_fe"], drop_first=True, dtype=float).reset_index(drop=True),
            pd.get_dummies(iv["year_fe"], drop_first=True, dtype=float).reset_index(drop=True)
        ], axis=1)
        exog = sm.add_constant(exog, has_constant="add")
        iv_res = IV2SLS(
            dependent=iv["SRD"].reset_index(drop=True),
            exog=exog,
            endog=iv[["GF"]].reset_index(drop=True),
            instruments=iv[["bank_financial_institutions"]].reset_index(drop=True)
        ).fit(cov_type="robust")
        save_summary(iv_res, "IV_2SLS")
        results["IV_2SLS"] = iv_res
    else:
        warnings.warn("Skipping IV model: bank_financial_institutions column is missing.")

    # Save compact coefficient table for model types with a common result interface.
    coefficient_table(results).to_csv(OUT / "panel_model_coefficients.csv", index=False)
    print("Panel models completed. See outputs/*_summary.txt and panel_model_coefficients.csv.")

if __name__ == "__main__":
    main()
