# Stage 1: NASA POWER daily precipitation (PRECTOTCORR) for 20 Indonesian locations -> monthly totals.
# The raw files are data/raw/nasa_power/<location>_daily_1981_2025.csv (daily API values, missing -999 stored as empty cells), kept exactly
# as shipped with the dataset, so their column headers are Indonesian: tanggal = date, lokasi = location, bulan = month.
# NASA POWER is a gridded estimate without missing days for 1981-2025, so every month is a valid training and evaluation target;
# the "ok" mask is kept for compatibility with the SILO pipeline and is True wherever all days of the month are valid.
#   python src/prepare_data.py
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA_RAW, DATA_PROCESSED, STATIONS_ALL, YEAR0

D_NASA = os.path.join(DATA_RAW, "nasa_power")


def prepare_rainfall(out_dir=DATA_PROCESSED, verbose=True):
    os.makedirs(out_dir, exist_ok=True)
    months = pd.period_range(f"{YEAR0}-01", "2025-12", freq="M"); yrs = np.array([m.year for m in months])
    out = {"months": [str(m) for m in months], "stations": {}}; rows = []
    loc = pd.read_csv(os.path.join(D_NASA, "locations.csv"), encoding="utf-8-sig").set_index("location")
    for st in STATIONS_ALL:
        d = pd.read_csv(os.path.join(D_NASA, f"{st}_daily_1981_2025.csv"), encoding="utf-8-sig", parse_dates=["tanggal"])
        d["ym"] = d["tanggal"].dt.to_period("M")
        g = d.groupby("ym").agg(rain=("PRECTOTCORR", "sum"), n_valid=("PRECTOTCORR", "count"), n_days=("PRECTOTCORR", "size")).reindex(months)
        rain = g["rain"].values.astype(float); ok = (g["n_valid"].values == g["n_days"].values) & (g["n_days"].values > 0)
        rain[~ok] = np.nan
        # reconcile with the monthly file shipped with the dataset (PRECTOTCORR_total_mm)
        mf = os.path.join(D_NASA, f"{st}_monthly_1981_2025.csv")
        if os.path.exists(mf):
            m = pd.read_csv(mf, encoding="utf-8-sig"); m.index = pd.PeriodIndex(m["bulan"], freq="M")
            diff = float(np.nanmax(np.abs(m["PRECTOTCORR_total_mm"].reindex(months).values - rain)))
            assert diff < 0.05, f"{st}: monthly totals differ from the shipped monthly file by {diff:.3f} mm"
        mon = (np.arange(len(months)) % 12) + 1
        clim = np.array([np.nanmean(rain[(mon == k) & ok]) for k in range(1, 13)])
        def cov(a, b): k = (yrs >= a) & (yrs <= b); return round(float(ok[k].mean()), 3)
        rows.append(dict(station=st, lat=float(loc.loc[st, "latitude"]), lon=float(loc.loc[st, "longitude"]), mean_mm=round(float(np.nanmean(rain)), 1),
                         wettest_month=int(np.argmax(clim) + 1), driest_month=int(np.argmin(clim) + 1), seasonality=round(float(clim.max() / max(clim.min(), 1e-9)), 2),
                         cov_1981_1995=cov(1981, 1995), cov_1996_2012=cov(1996, 2012), cov_2013_2017=cov(2013, 2017), cov_2018_2025=cov(2018, 2025)))
        out["stations"][st] = {"rain": [None if np.isnan(x) else float(x) for x in rain], "ok": ok.astype(int).tolist(),
                               "lat": float(loc.loc[st, "latitude"]), "lon": float(loc.loc[st, "longitude"])}
    summary = pd.DataFrame(rows)
    with open(os.path.join(out_dir, "monthly_20.json"), "w", newline="\n") as f: json.dump(out, f)
    summary.to_csv(os.path.join(out_dir, "station_summary.csv"), index=False, lineterminator="\n")
    if verbose: print(f"{len(STATIONS_ALL)} locations, {len(months)} months; all months valid:", bool(all(all(s['ok']) for s in out['stations'].values())))
    return summary


if __name__ == "__main__":
    print(prepare_rainfall().to_string(index=False))
