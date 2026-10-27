# Stage 5: harmonise the two per-network meta-datasets into one joint meta-dataset (data/joint/meta_joint.npz).
#   python src/harmonise.py
# Per meta-sample (station s, origin o, horizon h) stage 4 of each network stores E (12 x 11 scaled error history), P (11 member forecasts,
# scaled), y (scaled target), ok (valid target), SC (sigma_s), Mo (sin-cos of target month), Rg (6 regime features), H (one-hot horizon),
# T (target month index in the region's calendar), j (horizon index), st (station index in the region). Harmonisation:
#   * the 11-member pool is aligned by position (9 base models + expanding climatology + windowed climatology: 30 years AU, 20 years ID);
#   * the regime features are reduced to the three indices available in both regions (Nino3.4, DMI, SOI), each already standardised
#     with its region's pre-validation data (causal);
#   * the six static station features are recomputed from the raw monthly series (pre-validation data of each region only) and
#     standardised across all 35 stations;
#   * the time split (0 meta-training, 1 validation, 2 test) stays that of the region; station indices become global (0..34);
#   * rows keep the order of the per-network meta-datasets, so the prediction indices (idx) of the per-network gates map with a row offset.
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

os.makedirs(DATA_JOINT, exist_ok=True)
parts = []; stations = []; region_of = []; static_raw = []; offsets = {}; row0 = 0; src_pools = {}
for r in REGION_LIST:
    cfg = REGIONS[r]
    Z = np.load(os.path.join(region_results(r), "meta_dataset.npz"), allow_pickle=False)
    sts = [str(s) for s in Z["stations"]]; pool = [str(p) for p in Z["pool"]]; src_pools[r] = pool
    assert pool[:NB] == POOL[:NB] and pool[NB] == "Clim-expanding" and pool[NB + 1] == cfg["clim_window"], pool
    MON = json.load(open(os.path.join(region_processed(r), cfg["monthly"])))["stations"]
    n = len(Z["y"]); offsets[r] = row0
    T = Z["T"]; split = np.where(T < cfg["VA0"], 0, np.where(T < cfg["TE0"], 1, 2))
    parts.append(dict(E=Z["E"].astype(np.float32), P=Z["P"].astype(np.float32), y=Z["y"].astype(np.float32), ok=Z["ok"].astype(bool),
                      SC=Z["SC"].astype(np.float32), Mo=Z["Mo"].astype(np.float32), Rg=Z["Rg"][:, cfg["rg_cols"]].astype(np.float32),
                      H=Z["H"].astype(np.float32), T=T.astype(np.int32), j=Z["j"].astype(np.int32), split=split.astype(np.int32),
                      st=(Z["st"] + len(stations)).astype(np.int32), region=np.full(n, REGION_LIST.index(r), np.int32)))
    for st in sts:
        y = np.array(MON[st]["rain"], float); ok = np.array(MON[st]["ok"], bool); VA0 = cfg["VA0"]
        mon = (np.arange(cfg["N"]) % 12) + 1
        ann = np.array([y[:VA0][ok[:VA0] & (mon[:VA0] == m)].mean() for m in range(1, 13)])
        c1 = np.abs(np.sum(ann * np.exp(-2j * np.pi * np.arange(12) / 12))) / ann.sum()
        static_raw.append([np.sin(np.deg2rad(MON[st]["lat"])), np.cos(np.deg2rad(MON[st]["lon"])), np.sin(np.deg2rad(MON[st]["lon"])),
                           np.log(ann.sum()), c1, np.argmax(ann) / 11.0])
        stations.append(st); region_of.append(r)
    row0 += n
    print(r, ":", len(sts), "stations,", n, "meta-samples, split sizes", [int((split == s).sum()) for s in range(3)])
D = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
S = np.array(static_raw); S = (S - S.mean(0)) / (S.std(0) + 1e-9)
np.savez(os.path.join(DATA_JOINT, "meta_joint.npz"), stations=np.array(stations), region_of_station=np.array(region_of), static=S.astype(np.float32),
         pool=np.array(POOL), regime=np.array(REGIME), row_offset=np.array([offsets[r] for r in REGION_LIST]), **D)
print("joint meta-dataset:", len(D["y"]), "rows,", len(stations), "stations; Rg mean/std per region:")
for ri, r in enumerate(REGION_LIST):
    k = D["region"] == ri; print(" ", r, np.round(D["Rg"][k].mean(0), 3), np.round(D["Rg"][k].std(0), 3))
