"""Create a kernel-density figure for selected years from a real panel dataset."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from common import load_panel, OUT, require_columns

def main():
    df = load_panel()
    require_columns(df, ["year", "SRD"])
    selected = [2004, 2005, 2006, 2008, 2010, 2012, 2014, 2016, 2018, 2020, 2021, 2022]
    years = [y for y in selected if y in set(df["year"])]
    if not years:
        raise ValueError("No years from the paper's selected kernel-density figure were found.")
    grid = np.linspace(max(0, df["SRD"].min()), df["SRD"].max(), 300)
    plt.figure(figsize=(10, 6))
    for year in years:
        vals = df.loc[df["year"] == year, "SRD"].dropna().to_numpy()
        if len(vals) > 1 and np.std(vals) > 0:
            kde = gaussian_kde(vals)
            plt.plot(grid, kde(grid), label=str(year), linewidth=1.4)
    plt.xlabel("Sustainable rural development index (SRD)")
    plt.ylabel("Density")
    plt.title("Kernel density of sustainable rural development by year")
    plt.legend(ncol=3, fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(OUT / "srd_kernel_density.png", dpi=200)
    plt.close()
    print("Saved outputs/srd_kernel_density.png")

if __name__ == "__main__":
    main()
