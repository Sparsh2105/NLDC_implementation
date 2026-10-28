"""Moran's I and Spatial Durbin Model using coordinates and PySAL."""
import numpy as np
import pandas as pd
from scipy.sparse import identity
from scipy.sparse.linalg import spsolve
from libpysal.weights import KNN
from esda.moran import Moran
from spreg import ML_Lag
from common import load_panel, OUT, CONTROLS, require_columns

def make_weights(cross_section, k=5):
    if not {"longitude", "latitude"}.issubset(cross_section.columns):
        raise ValueError("Spatial analysis needs longitude and latitude columns.")
    coords = list(zip(cross_section["longitude"], cross_section["latitude"]))
    w = KNN.from_array(coords, k=min(k, len(coords)-1), ids=list(cross_section["city_id"]))
    w.transform = "R"
    return w

def impacts_sdm(beta, theta, rho, W):
    """Average direct, indirect, total impact for one covariate using SDM multiplier."""
    n = W.n
    I = identity(n, format="csc")
    Wsp = W.sparse.tocsc()
    A = I - rho * Wsp
    # The derivative matrix: (I-rho W)^(-1) (beta I + theta W)
    B = beta * I + theta * Wsp
    S = spsolve(A, B.toarray())
    direct = float(np.trace(S) / n)
    total = float(S.sum() / n)
    indirect = total - direct
    return {"direct": direct, "indirect": indirect, "total": total}

def main():
    df = load_panel()
    require_columns(df, ["SRD", "GF"] + CONTROLS + ["longitude", "latitude"])
    # A cross-sectional spatial model per year is not identical to a panel SDM.
    # This pooled setup includes year indicators and city-year rows; coordinates repeat.
    # For a valid city-level W matrix, estimate a selected year or city-average model.
    # Here, city averages are used for Moran's I and the SDM demonstration.
    avg = df.groupby(["city_id", "city_name", "province"], as_index=False).agg({
        "SRD": "mean", "GF": "mean", "URB": "mean", "UED": "mean",
        "HUC": "mean", "FDL": "mean", "OPE": "mean",
        "longitude": "first", "latitude": "first"
    }).dropna()
    w = make_weights(avg, k=5)

    moran_rows = []
    for var in ["SRD", "GF"]:
        mi = Moran(avg[var].to_numpy(), w, permutations=999)
        moran_rows.append({"variable": var, "moran_I": mi.I, "p_sim": mi.p_sim,
                           "z_sim": mi.z_sim})
    pd.DataFrame(moran_rows).to_csv(OUT / "moran_global.csv", index=False)

    y = avg[["SRD"]].to_numpy()
    x_cols = ["GF"] + CONTROLS
    X = avg[x_cols].to_numpy()
    # slx_lags=1 adds spatially lagged X terms, making this a Spatial Durbin specification.
    model = ML_Lag(
        y, X, w=w, slx_lags=1, method="full",
        name_y="SRD", name_x=x_cols, name_w="KNN row-standardized (k=5)"
    )
    with open(OUT / "spatial_durbin_summary.txt", "w", encoding="utf-8") as f:
        f.write(model.summary)

    # Try to calculate impacts only when PySAL exposes clearly matching GF and W_GF
    # names. The impact matrix is (I - rho W)^(-1) (beta I + theta W).
    names = [str(n) for n in getattr(model, "name_x", [])]
    params = np.asarray(model.betas).flatten()
    rho = float(np.asarray(model.rho).flatten()[0])
    gf_idx = next((i for i, n in enumerate(names) if n.upper() in {"GF", "X_GF"}), None)
    wgf_idx = next((i for i, n in enumerate(names) if n.upper() in {"W_GF", "WX_GF", "W_GF1"}), None)
    if gf_idx is not None and wgf_idx is not None and gf_idx < len(params) and wgf_idx < len(params):
        impact = impacts_sdm(float(params[gf_idx]), float(params[wgf_idx]), rho, w)
        pd.DataFrame([{"variable": "GF", **impact, "rho": rho}]).to_csv(
            OUT / "spatial_sdm_impacts.csv", index=False
        )
    else:
        with open(OUT / "spatial_sdm_impact_note.txt", "w", encoding="utf-8") as f:
            f.write(
                "Impact calculation skipped because this PySAL version did not expose "
                "unambiguous GF and W_GF coefficient names. Inspect spatial_durbin_summary.txt "
                "and model.name_x ordering before calculating impacts. Do not infer coefficient "
                "positions without checking the installed package version."
            )

    print("Spatial model completed. Review spatial_durbin_summary.txt and coefficient naming.")
    print("Moran's I results saved to outputs/moran_global.csv.")
    print("This city-average model is a demonstration; use the exact panel SDM and weight matrix from the paper for replication.")

if __name__ == "__main__":
    main()
