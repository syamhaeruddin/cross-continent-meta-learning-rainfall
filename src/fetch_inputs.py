# Stage 0: copy the required outputs of the two source repositories into data/input/<region>/.
# The source repositories are expected as sibling folders of this repository (clone them from GitHub first),
# or give their paths explicitly:
#   python src/fetch_inputs.py [path_to_australia_repo] [path_to_indonesia_repo]
# Files copied per region: results/meta_dataset.npz, results/pred_local_*.npz, results/pred_global_*.npz, results/pred_transfer_*.npz,
#                          results/evaluation_results.json, data/processed/<monthly json>
import os, sys, glob, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, DATA_INPUT, REGIONS, REGION_LIST

paths = dict(zip(REGION_LIST, sys.argv[1:3])) if len(sys.argv) >= 3 else {r: os.path.join(os.path.dirname(ROOT), REGIONS[r]["repo"]) for r in REGION_LIST}
for r in REGION_LIST:
    src = paths[r]; dst = os.path.join(DATA_INPUT, r); os.makedirs(dst, exist_ok=True)
    assert os.path.exists(os.path.join(src, "results", "meta_dataset.npz")), f"meta_dataset.npz not found in {src}/results — run 'python src/meta_gate.py build' there first"
    files = [os.path.join(src, "results", "meta_dataset.npz"), os.path.join(src, "results", "evaluation_results.json"),
             os.path.join(src, "data", "processed", REGIONS[r]["monthly"])]
    for pat in ["pred_local_*.npz", "pred_global_*.npz", "pred_transfer_*.npz"]: files += sorted(glob.glob(os.path.join(src, "results", pat)))
    for f in files:
        if os.path.exists(f): shutil.copy2(f, dst)
        else: print("missing (skipped):", f)
    print(r, "<-", src, ":", len(os.listdir(dst)), "files")
