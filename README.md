# Green Finance and Sustainable Rural Development — Reproducible Analysis Starter

This repository contains Python scripts to implement the main analysis families described in:

Yi, C., Xu, B., & Lin, K. (2025). *Exploring the Impact of Green Finance on Sustainable Rural Development: Evidence From 283 Cities in China*. Discrete Dynamics in Nature and Society, Article 6680364. https://doi.org/10.1155/ddns/6680364

## Important reproducibility note

The paper describes a panel of 283 Chinese prefecture-level cities from 2004–2022 (5,377 observations), with data from Chinese statistical yearbooks. The PDF does **not** provide the complete city-year raw dataset or all city boundary/geographic files. Therefore this repository is an implementation template, not a claim that the published coefficients have been independently reproduced. The scripts will calculate results from data you provide; they do not fabricate data or hard-code the paper's results.

## Included analyses

1. Build a Sustainable Rural Development (SRD) composite index using the paper's DPSIR indicator framework and entropy weighting.
2. Build a Green Finance (GF) index from six indicator groups.
3. Two-way fixed-effects baseline model.
4. Moderation models for environmental regulation (ER) and digital economy (DE).
5. Robustness checks: lagged SRD, winsorization, alternative GF index (PCA).
6. DID model for the 2017 Green Financial Reform and Innovation pilot policy.
7. IV/2SLS model using the number of bank financial institutions as an instrument.
8. Spatial Durbin Model (SDM), direct/indirect/total impacts, Moran's I, and a kernel-density plot.
9. Output tables and plots to `outputs/`.

## Setup

Recommended Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\\Scripts\\activate         # Windows

pip install -r requirements.txt
```

## Data files expected

### `data/panel_raw.csv`

One row per city-year. Required ID columns:

- `city_id`: stable city identifier
- `city_name`: city name
- `province`: province name
- `year`: integer year from 2004 to 2022

Required constructed variables for model scripts:

- `SRD`: sustainable rural development index (or generate it with `build_indices.py`)
- `GF`: green finance index (or generate it with `build_indices.py`)
- `ER`: environmental regulation index
- `DE`: digital economy index
- `URB`, `UED`, `HUC`, `FDL`, `OPE`: controls
- `bank_financial_institutions`: instrument candidate for 2SLS
- `longitude`, `latitude`: coordinates for spatial analysis, if available

The starter `data/panel_template.csv` documents the structure. It contains column headers only, not research data.

### Raw indicator columns for index construction

The script uses column lists declared near the top of `src/build_indices.py`. Replace the example column names with the exact names in your collected dataset. Use `+` for positive indicators and `-` for negative indicators. The DPSIR categories and indicator descriptions follow Table 1 in the paper; the six GF dimensions follow Table 2.

Do not run index construction until the raw indicator columns have been correctly mapped and their units/polarities checked.

## Run order

1. Put the licensed/collected raw data into `data/panel_raw.csv`.
2. Update indicator column names in `src/build_indices.py`.
3. Run:
   ```bash
   python src/build_indices.py
   python src/run_panel_models.py
   python src/run_spatial_models.py
   python src/make_figures.py
   ```
4. Inspect the generated CSV tables, model summaries, and figures. Report your own estimates, not the coefficients printed in the paper unless clearly labelled as published-paper results.

## Notes on methods

- The baseline is a city and year fixed-effects model.
- Moderation models include `GF × ER` and `GF × DE`.
- DID uses pilot provinces named in the paper: Zhejiang, Jiangxi, Guangdong, Guizhou, and Xinjiang, with post-policy years beginning in 2017. Province-name spelling in your dataset must match.
- The instrument strategy in the paper describes the number of bank financial institutions and reports 2SLS diagnostics. Instrument validity is an empirical assumption and must be discussed, not assumed.
- The SDM requires a defensible spatial weights matrix. This starter creates a row-standardized K-nearest-neighbour matrix from coordinates. If the original study used a different matrix, replace this with the paper's exact specification before claiming replication.
- Spatial impacts are computed using the SDM impact matrix formula. Treat estimates as model-dependent and verify them against the chosen spatial package/version.
- Kernel density plot is generated from city-year SRD values by selected years.
- A true map/standard-deviation ellipse needs valid coordinates and/or GIS boundary files. The current starter includes a kernel-density plot and spatial-model code; it does not fabricate a China map.

## Citation

Cite the paper in reports and presentations:
Yi, C., Xu, B., & Lin, K. (2025). *Exploring the Impact of Green Finance on Sustainable Rural Development: Evidence From 283 Cities in China*. https://doi.org/10.1155/ddns/6680364
