# Stage 7: evaluation of the joint, region-only, and leave-region-out gates against the per-network gates and static benchmarks of each region;
# per-station DM tests (Holm within region), panel DM tests per region (squared and absolute errors), scaling curve, weight composition.
# Writes results/evaluation_results.json.
#   python src/evaluate.py
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from scipy.optimize import nnls

D = load_joint(); sts = D["stations"]; pool = [str(p) for p in D["pool"]]; M = len(pool)
P, y, SC, T, J, ST, RG = D["P"], D["y"], D["SC"], D["T"], D["j"], D["st"], D["region"]
n = len(y); split = D["split"]; offs = {r: int(D["row_offset"][i]) for i, r in enumerate(REGION_LIST)}
region_sts = {r: [s for s, rr in zip(sts, D["region_of_station"]) if rr == r] for r in REGION_LIST}
eval_sts = {r: [s for s in region_sts[r] if s not in REGIONS[r]["exclude"]] for r in REGION_LIST}

# ---- static benchmarks, recomputed from the joint meta-dataset (same definitions as in the per-network studies)
PRED = {"Clim-window": P[:, M - 1].astype(float), "Clim-expanding": P[:, M - 2].astype(float), "Mean": P[:, :NB].mean(1).astype(float)}
NN = np.full(n, np.nan); best = np.full(n, np.nan); W_NNLS = {}
for si in range(len(sts)):
    for j in range(4):
        k = (ST == si) & (J == j); kp = k & (split < 2) & D["ok"]
        w, _ = nnls(P[kp], y[kp]); W_NNLS[(si, j)] = w; NN[k] = P[k] @ w
        best[k] = P[k, int(np.argmin(((P[kp] - y[kp, None])**2).mean(0)))]
PRED["NNLS"] = NN; PRED["Best model"] = best

def put(name, idx, arr):
    full = PRED.setdefault(name, np.full(n, np.nan)); full[idx] = arr.mean(0)
# ---- per-network gates from stage 4 (row indices shifted by the region offset)
for r in REGION_LIST:
    din = region_results(r); o = offs[r]
    for f in glob.glob(os.path.join(din, "pred_local_*.npz")):
        Z = np.load(f)
        for nm in ["L-G3", "L-G4"]: put(nm, Z["idx"] + o, Z[nm])
    for nm in ["GG3", "GG4"]:
        f = os.path.join(din, f"pred_global_{nm}.npz")
        if os.path.exists(f): Z = np.load(f); put(nm, Z["idx"] + o, Z["pred"])
    for f in glob.glob(os.path.join(din, "pred_transfer_*.npz")):
        Z = np.load(f); put("GG4-T", Z["idx"] + o, Z["pred"])
# ---- gates of this experiment
for r in REGION_LIST:
    f = os.path.join(OUT, f"pred_rg_{r}.npz")
    if os.path.exists(f): Z = np.load(f); put("RG", Z["idx"], Z["pred"])
    f = os.path.join(OUT, f"pred_lro_{r}.npz")
    if os.path.exists(f): Z = np.load(f); put("LRO", Z["idx"], Z["pred"])   # trained on r, predicts the other region
for nm in ["JG-E", "JG-S"]:
    f = os.path.join(OUT, f"pred_jg_{nm}.npz")
    if os.path.exists(f): Z = np.load(f); put(nm, Z["idx"], Z["pred"])
METHODS = [m for m in ["Clim-window", "Clim-expanding", "Mean", "NNLS", "Best model", "L-G3", "L-G4", "GG3", "GG4", "GG4-T", "RG", "JG-S", "JG-E", "LRO"] if m in PRED]

def mask(si, j, a, b): return (ST == si) & (J == j) & (split == 2) & D["ok"] & (T >= a) & (T < b)

R = {"regions": REGION_LIST, "stations": eval_sts, "methods": METHODS, "nse": {}, "nse_mean": {}, "dm": {}, "wins": {}, "panel": {}, "weights": {}, "scaling": {}, "epochs": {}}
PAIRS = [("JG-E", "RG"), ("JG-E", "GG3"), ("RG", "GG3"), ("JG-S", "RG"), ("JG-S", "JG-E"), ("JG-E", "L-G3"), ("JG-E", "NNLS"), ("JG-E", "Mean"), ("JG-E", "Clim-window"),
         ("LRO", "L-G3"), ("LRO", "GG4-T"), ("LRO", "GG3"), ("LRO", "NNLS"), ("LRO", "Mean"), ("LRO", "Clim-window"), ("LRO", "L-G4"), ("JG-S", "L-G3"), ("JG-S", "NNLS")]
PAIRS = [p for p in PAIRS if p[0] in PRED and p[1] in PRED]
for r in REGION_LIST:
    cfg = REGIONS[r]; PER = cfg["periods"]; main = list(PER)[0]
    R["nse"][r] = {}; R["nse_mean"][r] = {}; R["dm"][r] = {}; R["wins"][r] = {}; R["panel"][r] = {}
    for per, (a, b) in PER.items():
        R["nse"][r][per] = {}
        for st in eval_sts[r]:
            si = sts.index(st); R["nse"][r][per][st] = {}
            for j, h in enumerate(HORIZONS):
                k = mask(si, j, a, b)
                if k.sum() < 12: continue
                yy = y[k] * SC[k]; row = {"n": int(k.sum())}
                for m in METHODS:
                    if np.isnan(PRED[m][k]).any(): continue
                    mm = metrics(yy, PRED[m][k] * SC[k]); row[m] = mm["NSE"]; row[m + "_RMSE"] = mm["RMSE"]; row[m + "_MAE"] = mm["MAE"]
                R["nse"][r][per][st][h] = row
        R["nse_mean"][r][per] = {}
        for h in HORIZONS:
            common_st = [s for s in eval_sts[r] if h in R["nse"][r][per][s] and all(m in R["nse"][r][per][s][h] for m in METHODS)]
            R["nse_mean"][r][per][h] = {"stations": len(common_st), **{m: float(np.mean([R["nse"][r][per][s][h][m] for s in common_st])) for m in METHODS}}
    for a_, b_ in PAIRS:
        key = f"{a_} vs {b_}"; rows = []; a, b = PER[main]
        for st in eval_sts[r]:
            si = sts.index(st)
            for j, h in enumerate(HORIZONS):
                k = mask(si, j, a, b)
                if k.sum() < 12 or np.isnan(PRED[a_][k]).any() or np.isnan(PRED[b_][k]).any(): continue
                e1, e2 = y[k] - PRED[a_][k], y[k] - PRED[b_][k]; t = dm_test(e1, e2, h)
                rows.append(dict(st=st, h=h, DM=t["DM"], p=t["p"], dMSE=float(1 - np.mean(e1**2) / np.mean(e2**2))))
        if not rows: continue
        for row, q in zip(rows, holm([x["p"] for x in rows])): row["p_holm"] = float(q)
        R["dm"][r][key] = rows
        nb = int(sum(x["dMSE"] > 0 for x in rows))
        R["wins"][r][key] = {"better": nb, "total": len(rows), "median_dMSE_pct": float(100 * np.median([x["dMSE"] for x in rows])),
                             "mean_dMSE_pct": float(100 * np.mean([x["dMSE"] for x in rows])),
                             "significantly better (Holm)": int(sum((x["DM"] < 0) & (x["p_holm"] < 0.05) for x in rows)),
                             "significantly worse (Holm)": int(sum((x["DM"] > 0) & (x["p_holm"] < 0.05) for x in rows)),
                             "sign test p": float(stats.binomtest(nb, len(rows), 0.5).pvalue)}
        # panel DM per region: loss differentials averaged across stations at each target month; Holm over the 4 horizons
        R["panel"][r][key] = {}
        for per, (a, b) in PER.items():
            for loss in ["squared", "absolute"]:
                keys = []
                for j, h in enumerate(HORIZONS):
                    dd = {}
                    for st in eval_sts[r]:
                        si = sts.index(st); k = np.where(mask(si, j, a, b))[0]
                        if len(k) < 12 or np.isnan(PRED[a_][k]).any() or np.isnan(PRED[b_][k]).any(): continue
                        e1, e2 = y[k] - PRED[a_][k], y[k] - PRED[b_][k]
                        dl = (e1**2 - e2**2) if loss == "squared" else (np.abs(e1) - np.abs(e2))
                        for kk, v in zip(k, dl): dd.setdefault(int(T[kk]), []).append(v)
                    Ts = sorted(dd); d = np.array([np.mean(dd[t]) for t in Ts])
                    if len(d) < 12: continue
                    t = dm_stat(d, h); kk = f"{per} {loss} h={h}"; keys.append(kk)
                    R["panel"][r][key][kk] = {"DM": t["DM"], "p": t["p"], "months": len(d), "mean_dloss_scaled": float(d.mean())}
                for kk, q in zip(keys, holm([R["panel"][r][key][kk]["p"] for kk in keys])): R["panel"][r][key][kk]["p_holm"] = float(q)

# ---- weight composition of the joint gate JG-E per region and horizon (model families)
FAM = {"Statistical": ["SARIMA"], "Machine learning": ["SVR", "XGBoost"], "Deep learning": ["LSTM", "GRU"],
       "Hybrid": ["XGBoost-VMD", "XGBoost-WT", "LSTM-VMD", "LSTM-WT"], "Climatology": ["Clim-expanding", "Clim-window"]}
for nm, fn in [("JG-E", "pred_jg_JG-E.npz"), ("JG-S", "pred_jg_JG-S.npz")]:
    f = os.path.join(OUT, fn)
    if not os.path.exists(f): continue
    Z = np.load(f); idx, Wt = Z["idx"], Z["w"]; R["weights"][nm] = {}
    for ri, r in enumerate(REGION_LIST):
        R["weights"][nm][r] = {}
        ev = np.isin(ST[idx], [sts.index(s) for s in eval_sts[r]])
        for j, h in enumerate(HORIZONS):
            k = ev & (J[idx] == j)
            R["weights"][nm][r][h] = {fm: float(Wt[k][:, [pool.index(x) for x in FAM[fm]]].sum(1).mean()) for fm in FAM}
            R["weights"][nm][r][f"{h} members"] = dict(zip(pool, Wt[k].mean(0).round(3).tolist()))
for f in sorted(glob.glob(os.path.join(OUT, "pred_*.npz"))):
    Z = np.load(f)
    if "epochs" in Z: R["epochs"][os.path.basename(f)] = Z["epochs"].tolist()

# ---- scaling curve: holdout stations (3 AU + 4 ID), gates trained on k stations; mean NSE and MSE change vs the stored local gate L-G3
hold = [s for r in REGION_LIST for s in REGIONS[r]["holdout"]]
def holdout_scores(pred):
    out = {}
    for h_i, h in enumerate(HORIZONS):
        v, dm = [], []
        for st in hold:
            r = D["region_of_station"][sts.index(st)]; a, b = list(REGIONS[r]["periods"].values())[0]; k = mask(sts.index(st), h_i, a, b)
            if np.isnan(pred[k]).any() or np.isnan(PRED["L-G3"][k]).any(): continue
            v.append(metrics(y[k] * SC[k], pred[k] * SC[k])["NSE"]); dm.append(1 - np.mean((y[k] - pred[k])**2) / np.mean((y[k] - PRED["L-G3"][k])**2))
        out[h] = {"NSE": float(np.mean(v)), "dMSE_vs_LG3_pct": float(100 * np.mean(dm)), "stations": len(v)}
    out["all"] = {"NSE": float(np.mean([out[h]["NSE"] for h in HORIZONS])), "dMSE_vs_LG3_pct": float(np.mean([out[h]["dMSE_vs_LG3_pct"] for h in HORIZONS]))}
    return out
for f in sorted(glob.glob(os.path.join(OUT, "pred_scale_k*_d*.npz"))):
    Z = np.load(f); pr = np.full(n, np.nan); pr[Z["idx"]] = Z["pred"].mean(0)
    nm = os.path.basename(f)[len("pred_scale_"):-4]; k_, d_ = nm.split("_")
    R["scaling"][nm] = {"k": int(k_[1:]), "draw": int(d_[1:]), "train_stations": [sts[i] for i in Z["train_stations"]], **holdout_scores(pr)}
for nm in ["L-G3", "GG4-T", "JG-S", "JG-E", "LRO", "NNLS"]:
    if nm in PRED: R["scaling"][f"ref {nm}"] = holdout_scores(PRED[nm])
json.dump(R, open(os.path.join(OUT, "evaluation_results.json"), "w", newline="\n"), indent=1)

# ---- printed summary
for r in REGION_LIST:
    main = list(REGIONS[r]["periods"])[0]
    print(f"\n== {REGIONS[r]['label']}: mean NSE across {len(eval_sts[r])} stations, {main}")
    for h in HORIZONS:
        v = R["nse_mean"][r][main][h]; print(f"h={h}", {m: round(v[m], 3) for m in METHODS})
    print("-- wins / median dMSE%")
    for k, v in R["wins"][r].items(): print(f"  {k}: {v['better']}/{v['total']}, median {v['median_dMSE_pct']:.1f}%")
    print("-- panel DM (squared), p Holm over horizons,", main)
    for k, v in R["panel"][r].items():
        print(f"  {k}:", [round(v[f'{main} squared h={h}']['p_holm'], 3) for h in HORIZONS if f'{main} squared h={h}' in v])
print("\n== scaling curve (holdout stations):")
for k in [4, 8, 14, 28]:
    vals = [v["all"]["NSE"] for nm, v in R["scaling"].items() if nm.startswith(f"k{k}_")]
    if vals: print(f"k={k}: NSE {np.mean(vals):.3f} ± {np.std(vals):.3f} ({len(vals)} draws)")
for nm in ["ref L-G3", "ref GG4-T", "ref JG-S", "ref LRO"]:
    if nm in R["scaling"]: print(nm, round(R["scaling"][nm]["all"]["NSE"], 3))
