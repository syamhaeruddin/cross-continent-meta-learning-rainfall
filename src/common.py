# Constants, file locations, and shared functions for the cross-network gate experiment.
# The whole pipeline runs inside this repository, from the raw rainfall records to the reported tables and figures:
#   stages 1-4 per network, under regions/<region>/ (src/run_regions.py), for
#     Australia : SILO rain gauges, 15 stations, 1926-2025
#     Indonesia : NASA POWER grid, 20 locations, 1981-2025
#   stages 5-8 on the pooled meta-samples of both networks (src/harmonise.py, gates.py, evaluate.py, report.py).
import os
for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"]:
    os.environ[v] = "1"
os.environ["TF_DETERMINISTIC_OPS"] = "1"; os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"; os.environ["PYTHONHASHSEED"] = "0"; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings; warnings.filterwarnings("ignore")
import json, numpy as np
from scipy import stats

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
REGIONS_DIR = os.path.join(ROOT, "regions")          # per-network data and results (stages 1-4)
DATA_JOINT = os.path.join(ROOT, "data", "joint")     # harmonised joint meta-dataset (stage 5)
OUT = os.environ.get("OUT_DIR", os.path.join(ROOT, "results")); os.makedirs(OUT, exist_ok=True)
REPORTS = os.path.join(ROOT, "reports")
HORIZONS = [1, 3, 6, 12]

# Region settings. Month indices (N, VA0, TE0, SPLIT) refer to each network's own calendar.
REGIONS = {
 "australia": dict(label="Australia (SILO)", monthly="monthly_15.json",
                   year0=1926, N=1200, VA0=1008, TE0=1068, SPLIT=1140, clim_window="Clim-30yr",
                   rg_cols=[0, 1, 2],                       # Rg of the network meta-dataset: [nino34, dmi, soi, nino34 3-mo mean, nino34 trend, dmi 3-mo mean]
                   exclude=["Coffs_Harbour", "Melbourne"],  # no observed months in the test period
                   periods={"2015-2025": (1068, 1200), "2015-2020": (1068, 1140), "2021-2025": (1140, 1200)},
                   holdout=["Darwin", "Brisbane", "Woomera"]),                   # transfer fold 0 of the Australian network
 "indonesia": dict(label="Indonesia (NASA POWER)", monthly="monthly_20.json",
                   year0=1981, N=540, VA0=384, TE0=444, SPLIT=492, clim_window="Clim-20yr",
                   rg_cols=[0, 2, 3],                       # Rg of the network meta-dataset: [nino34, nino4, dmi, soi, rmm1, rmm2]
                   exclude=[],
                   periods={"2018-2025": (444, 540), "2018-2021": (444, 492), "2022-2025": (492, 540)},
                   holdout=["Banda_Aceh", "Jakarta", "Banjarmasin", "Ambon"]),  # transfer fold 0 of the Indonesian network
}
REGION_LIST = ["australia", "indonesia"]
REGIME = ["nino34", "dmi", "soi"]      # the three indices available in both regions
POOL = ["SARIMA", "SVR", "XGBoost", "XGBoost-VMD", "XGBoost-WT", "LSTM", "GRU", "LSTM-VMD", "LSTM-WT", "Clim-expanding", "Clim-window"]
NB = 9                                 # number of base models (the two climatologies follow)


def region_root(region): return os.path.join(REGIONS_DIR, region)
def region_results(region): return os.path.join(REGIONS_DIR, region, "results")
def region_processed(region): return os.path.join(REGIONS_DIR, region, "data", "processed")


def metrics(a, p):
    a, p = np.asarray(a), np.asarray(p); e = a - p
    return {"MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e**2))), "NSE": float(1 - np.sum(e**2) / np.sum((a - a.mean())**2))}
def dm_stat(d, h):
    """Diebold-Mariano statistic with the Harvey-Leybourne-Newbold correction for a loss-differential series d; long-run variance up to lag h-1."""
    n = len(d); dc = d - d.mean()
    gam = [np.sum(dc[k:] * dc[:n - k]) / n for k in range(h)]
    v = gam[0] + 2 * sum(gam[1:]); v = v if v > 0 else gam[0]
    stat = d.mean() / np.sqrt(v / n) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return {"DM": float(stat), "p": float(2 * (1 - stats.t.cdf(abs(stat), df=n - 1)))}
def dm_test(e1, e2, h=1, loss="squared"):
    d = (e1**2 - e2**2) if loss == "squared" else (np.abs(e1) - np.abs(e2)); return dm_stat(d, h)
def holm(pvals):
    p = np.asarray(pvals, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj


def load_joint():
    Z = np.load(os.path.join(DATA_JOINT, "meta_joint.npz"), allow_pickle=False)
    D = {k: Z[k] for k in Z.files}
    D["stations"] = [str(s) for s in D["stations"]]; D["region_of_station"] = [str(s) for s in D["region_of_station"]]
    return D
