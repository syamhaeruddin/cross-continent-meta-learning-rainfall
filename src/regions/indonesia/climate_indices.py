# Stage 2: monthly climate indices 1981-2025 relevant to Indonesian rainfall.
# Source: the package data/raw/indices/ (file indeks_iklim_bulanan_1981_2025.csv, assembled from NOAA CPC, NOAA PSL, CRU, and BoM;
# URLs and hashes in metadata.json). Columns of the source file (Indonesian headers): bulan = month,
#   nino34_anom_C, nino4_anom_C : ERSST v5 SST anomalies (C) relative to the CPC 1991-2020 baseline
#   dmi_hadisst_C               : Dipole Mode Index from HadISST (NOAA PSL)
#   soi_cru                     : Southern Oscillation Index, CRU normalisation
#   rmm1_mean, rmm2_mean        : monthly means of the daily BoM MJO RMM1/RMM2 indices (Wheeler & Hendon, 2004)
# The Nino anomalies are re-referenced here to the 1981-1995 climatology so that no post-1995 information enters the baseline.
#   python src/climate_indices.py
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA_RAW, DATA_PROCESSED, YEAR0, INDICES

D_IDX = os.path.join(DATA_RAW, "indices")
COLS = {"nino34": "nino34_anom_C", "nino4": "nino4_anom_C", "dmi": "dmi_hadisst_C", "soi": "soi_cru", "rmm1": "rmm1_mean", "rmm2": "rmm2_mean"}


def compute_indices(out_dir=DATA_PROCESSED, verbose=True):
    months = pd.period_range(f"{YEAR0}-01", "2025-12", freq="M")
    src = pd.read_csv(os.path.join(D_IDX, "indeks_iklim_bulanan_1981_2025.csv"), encoding="utf-8-sig")
    src.index = pd.PeriodIndex(src["bulan"], freq="M"); src = src.reindex(months)
    df = pd.DataFrame({k: src[v].values.astype(float) for k, v in COLS.items()}, index=months)
    for c in ["nino34", "nino4"]:   # re-reference the anomalies to the 1981-1995 climatology (causal baseline)
        base = df.loc[(df.index.year >= 1981) & (df.index.year <= 1995), c].groupby(lambda p: p.month).mean()
        df[c] = df[c] - np.array([base[p.month] for p in months])
    df = df[INDICES]; df.index = months.astype(str)
    os.makedirs(out_dir, exist_ok=True); df.to_csv(os.path.join(out_dir, "climate_indices_1981_2025.csv"), lineterminator="\n")
    if verbose:
        print("missing values:", df.isna().sum().to_dict())
        print("correlation between indices:"); print(df.corr().round(2).to_string())
    return df


if __name__ == "__main__":
    print(compute_indices().describe().round(2).to_string())
