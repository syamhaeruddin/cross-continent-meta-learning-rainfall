# Stages 1-4, run separately for each station network from the raw data in regions/<region>/data/raw/:
#   1 rainfall preprocessing, 2 climate indices, 3 base-model pool with rolling retraining,
#   4 meta-dataset and the per-network gates (local L-G3/L-G4, global GG3/GG4, transfer GG4-T) used later as benchmarks.
# The scripts under src/regions/<region>/ are those of the published single-network studies; they read and write under
# regions/<region>/ through the environment variable REGION_ROOT.
#   python src/run_regions.py                  -> both networks, all stages
#   python src/run_regions.py australia        -> one network
#   python src/run_regions.py indonesia meta   -> one network, one phase (data | base | meta | all)
# Existing outputs are skipped; delete regions/<region>/results/ to retrain from scratch.
# Parallel processes: environment variable WORKERS (default 8).
import os, sys, subprocess, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, SRC, REGION_LIST, region_root

PY = sys.executable
WORKERS = os.environ.get("WORKERS", "8")


def run_region(region, phase="all"):
    rsrc = os.path.join(SRC, "regions", region); rroot = region_root(region)
    for d in ["data/processed", "results", "log"]: os.makedirs(os.path.join(rroot, d), exist_ok=True)
    env = {**os.environ, "REGION_ROOT": rroot, "WORKERS": WORKERS, "PYTHONIOENCODING": "utf-8"}
    steps = []
    if phase in ("data", "all"): steps += [["prepare_data.py"], ["climate_indices.py"]]
    if phase in ("base", "all"): steps += [["run_all.py", "base"]]
    if phase in ("meta", "all"): steps += [["run_all.py", "meta"]]
    for st in steps:
        t = time.time(); print(f"\n=== {region}: {' '.join(st)}", flush=True)
        r = subprocess.run([PY, os.path.join(rsrc, st[0])] + st[1:], cwd=rroot, env=env)
        print(f"=== {region}: {st[0]} exit {r.returncode}, {round((time.time() - t) / 60, 1)} min", flush=True)
        if r.returncode != 0: sys.exit(r.returncode)


if __name__ == "__main__":
    args = sys.argv[1:]
    regions = [a for a in args if a in REGION_LIST] or REGION_LIST
    phase = next((a for a in args if a in ("data", "base", "meta", "all")), "all")
    t0 = time.time()
    for r in regions: run_region(r, phase)
    print("\nstages 1-4 done,", round((time.time() - t0) / 60, 1), "min total", flush=True)
