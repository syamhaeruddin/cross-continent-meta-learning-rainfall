# Stage 8: tables (CSV) and figures (PNG) of the experiment, from results/evaluation_results.json.
#   python src/report.py
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
R = json.load(open(os.path.join(OUT, "evaluation_results.json"))); os.makedirs(REPORTS, exist_ok=True)
SHOW = [m for m in ["Clim-window", "Mean", "NNLS", "L-G3", "GG3", "GG4-T", "RG", "JG-S", "JG-E", "LRO"] if m in R["methods"]]
LAB = {"australia": "Australia (SILO, 13 stations)", "indonesia": "Indonesia (NASA POWER, 20 locations)"}

# Table 1: mean NSE per region, period, horizon
rows = []
for r in REGION_LIST:
    for per, byh in R["nse_mean"][r].items():
        for h in HORIZONS:
            v = byh[str(h)]; rows.append({"region": r, "period": per, "horizon": h, "stations": v["stations"], **{m: round(v[m], 3) for m in SHOW}})
        rows.append({"region": r, "period": per, "horizon": "mean", "stations": v["stations"], **{m: round(np.mean([byh[str(h)][m] for h in HORIZONS]), 3) for m in SHOW}})
T1 = pd.DataFrame(rows); T1.to_csv(os.path.join(REPORTS, "table1_mean_nse.csv"), index=False)

# Table 2: hypothesis tests per region (wins, median MSE change, panel DM p Holm per horizon, squared and absolute errors)
rows = []
for r in REGION_LIST:
    main = list(REGIONS[r]["periods"])[0]
    for key, w in R["wins"][r].items():
        pn = R["panel"][r][key]
        rows.append({"region": r, "comparison": key, "wins": f"{w['better']}/{w['total']}", "median_dMSE_pct": round(w["median_dMSE_pct"], 2),
                     "sig_better_holm": w["significantly better (Holm)"], "sig_worse_holm": w["significantly worse (Holm)"],
                     **{f"p_sq_h{h}": round(pn[f"{main} squared h={h}"]["p_holm"], 4) for h in HORIZONS if f"{main} squared h={h}" in pn},
                     **{f"p_abs_h{h}": round(pn[f"{main} absolute h={h}"]["p_holm"], 4) for h in HORIZONS if f"{main} absolute h={h}" in pn},
                     **{f"DM_sq_h{h}": round(pn[f"{main} squared h={h}"]["DM"], 2) for h in HORIZONS if f"{main} squared h={h}" in pn}})
T2 = pd.DataFrame(rows); T2.to_csv(os.path.join(REPORTS, "table2_tests.csv"), index=False)

# Table 3: scaling curve
rows = []
for k in [4, 8, 14, 28]:
    vs = [v for nm, v in R["scaling"].items() if nm.startswith(f"k{k}_")]
    if not vs: continue
    rows.append({"k_stations": k, "draws": len(vs), "NSE_mean": round(np.mean([v["all"]["NSE"] for v in vs]), 4), "NSE_sd": round(np.std([v["all"]["NSE"] for v in vs]), 4),
                 "dMSE_vs_LG3_pct": round(np.mean([v["all"]["dMSE_vs_LG3_pct"] for v in vs]), 2),
                 **{f"NSE_h{h}": round(np.mean([v[str(h)]["NSE"] for v in vs]), 4) for h in HORIZONS}})
for nm in ["ref L-G3", "ref NNLS", "ref GG4-T", "ref JG-S", "ref JG-E", "ref LRO"]:
    if nm in R["scaling"]:
        v = R["scaling"][nm]; rows.append({"k_stations": nm, "draws": "", "NSE_mean": round(v["all"]["NSE"], 4), "NSE_sd": "", "dMSE_vs_LG3_pct": round(v["all"]["dMSE_vs_LG3_pct"], 2),
                                           **{f"NSE_h{h}": round(v[str(h)]["NSE"], 4) for h in HORIZONS}})
T3 = pd.DataFrame(rows); T3.to_csv(os.path.join(REPORTS, "table3_scaling.csv"), index=False)

# Table 4: JG-E weight composition per region and horizon
rows = []
for nm in R["weights"]:
    for r in REGION_LIST:
        for h in HORIZONS:
            rows.append({"gate": nm, "region": r, "horizon": h, **{f: round(v, 3) for f, v in R["weights"][nm][r][str(h)].items()}})
T4 = pd.DataFrame(rows); T4.to_csv(os.path.join(REPORTS, "table4_weights.csv"), index=False)

# Figure 1: mean NSE per horizon, two panels
fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
STY = {"Clim-window": ("0.6", ":"), "Mean": ("0.4", "--"), "NNLS": ("#8d6e63", "--"), "L-G3": ("#1976d2", "-"), "GG3": ("#0d47a1", "-"), "GG4-T": ("#7b1fa2", "-"),
       "RG": ("#43a047", "-."), "JG-S": ("#fb8c00", "-"), "JG-E": ("#e53935", "-"), "LRO": ("#000000", "-")}
for ax, r in zip(axs, REGION_LIST):
    main = list(REGIONS[r]["periods"])[0]; byh = R["nse_mean"][r][main]
    for m in SHOW:
        c, ls = STY.get(m, ("k", "-")); ax.plot(range(4), [byh[str(h)][m] for h in HORIZONS], ls, color=c, marker="o", ms=3, label=m, lw=1.6 if m in ("JG-E", "LRO") else 1.1)
    ax.set_xticks(range(4)); ax.set_xticklabels([f"h={h}" for h in HORIZONS]); ax.set_ylabel("mean NSE"); ax.set_title(f"({'ab'[REGION_LIST.index(r)]}) {LAB[r]}, test {main}", fontsize=8.5); ax.grid(alpha=0.3)
axs[1].legend(fontsize=6.5, ncol=2, loc="lower left"); fig.tight_layout(); fig.savefig(os.path.join(REPORTS, "figure1_nse_by_horizon.png"), dpi=200); plt.close(fig)

# Figure 2: MSE change (%) of LRO vs L-G3 per station and horizon, per region
fig, axs = plt.subplots(1, 2, figsize=(9, 4.4), gridspec_kw={"width_ratios": [13, 20]})
fig.suptitle("Leave-region-out gate (trained on the other network) vs local gate L-G3: MSE reduction (%)", fontsize=8.5)
for ax, r in zip(axs, REGION_LIST):
    key = "LRO vs L-G3"; rows_ = R["dm"][r].get(key, []); stl = R["stations"][r]
    Mx = np.full((4, len(stl)), np.nan); Sg = np.zeros((4, len(stl)), bool)
    for x in rows_: Mx[HORIZONS.index(x["h"]), stl.index(x["st"])] = 100 * x["dMSE"]; Sg[HORIZONS.index(x["h"]), stl.index(x["st"])] = x["p"] < 0.05
    im = ax.imshow(Mx, cmap="RdBu", vmin=-15, vmax=15, aspect="auto")
    for i in range(4):
        for jj in range(len(stl)):
            if not np.isnan(Mx[i, jj]): ax.text(jj, i, f"{Mx[i, jj]:.0f}{'*' if Sg[i, jj] else ''}", ha="center", va="center", fontsize=6)
    ax.set_xticks(range(len(stl))); ax.set_xticklabels([s.replace("_", " ") for s in stl], rotation=90, fontsize=6.5); ax.set_yticks(range(4)); ax.set_yticklabels([f"h={h}" for h in HORIZONS])
    ax.set_title(f"({'ab'[REGION_LIST.index(r)]}) {LAB[r]}", fontsize=8)
fig.colorbar(im, ax=axs, shrink=0.8, label="MSE reduction (%)"); fig.savefig(os.path.join(REPORTS, "figure2_lro_vs_local_heatmap.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

# Figure 3: scaling curve
fig, ax = plt.subplots(figsize=(4.8, 3.4))
ks = [k for k in [4, 8, 14, 28] if any(nm.startswith(f"k{k}_") for nm in R["scaling"])]
for k in ks:
    vs = [v["all"]["NSE"] for nm, v in R["scaling"].items() if nm.startswith(f"k{k}_")]; ax.scatter([k] * len(vs), vs, color="#e53935", s=12, alpha=0.6)
ax.plot(ks, [np.mean([v["all"]["NSE"] for nm, v in R["scaling"].items() if nm.startswith(f"k{k}_")]) for k in ks], "-o", color="#e53935", label="transfer gate trained on k stations (mean of draws)")
for nm, c, ls in [("ref L-G3", "#1976d2", "--"), ("ref GG4-T", "#7b1fa2", "-.")]:
    if nm in R["scaling"]: ax.axhline(R["scaling"][nm]["all"]["NSE"], color=c, ls=ls, lw=1, label=nm[4:] + " (stored, same holdout stations)")
ax.set_xlabel("number of pooled training stations k"); ax.set_ylabel("mean NSE, 7 holdout stations, 4 horizons"); ax.set_xticks(ks); ax.grid(alpha=0.3); ax.legend(fontsize=6.5)
fig.tight_layout(); fig.savefig(os.path.join(REPORTS, "figure3_scaling_curve.png"), dpi=200); plt.close(fig)

# Figure 4: JG-E weight composition per family, region x horizon
if "JG-E" in R["weights"]:
    FAMS = list(R["weights"]["JG-E"]["australia"]["1"].keys()); cols = ["#1976d2", "#43a047", "#fb8c00", "#8e24aa", "0.6"]
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    for ax, r in zip(axs, REGION_LIST):
        bottom = np.zeros(4)
        for f, c in zip(FAMS, cols):
            v = np.array([R["weights"]["JG-E"][r][str(h)][f] for h in HORIZONS]); ax.bar(range(4), v, bottom=bottom, color=c, label=f, width=0.6); bottom += v
        ax.set_xticks(range(4)); ax.set_xticklabels([f"h={h}" for h in HORIZONS]); ax.set_title(f"({'ab'[REGION_LIST.index(r)]}) JG-E weights, {LAB[r]}", fontsize=8.5)
    axs[0].set_ylabel("mean weight"); axs[1].legend(fontsize=6.5, loc="upper right"); fig.tight_layout(); fig.savefig(os.path.join(REPORTS, "figure4_weights_by_family.png"), dpi=200); plt.close(fig)

# Summary text for the README
L = []
for r in REGION_LIST:
    main = list(REGIONS[r]["periods"])[0]; byh = R["nse_mean"][r][main]
    L.append(f"## {LAB[r]}, test {main}\n\n| Horizon | " + " | ".join(SHOW) + " |\n|---|" + "---|" * len(SHOW))
    for h in HORIZONS: L.append(f"| {h} | " + " | ".join(f"{byh[str(h)][m]:.3f}" for m in SHOW) + " |")
    L.append("| mean | " + " | ".join(f"{np.mean([byh[str(h)][m] for h in HORIZONS]):.3f}" for m in SHOW) + " |\n")
    L.append("| Comparison | Wins | Median MSE change | Panel p (Holm), h = 1 / 3 / 6 / 12, squared | absolute |\n|---|---|---|---|---|")
    for key, w in R["wins"][r].items():
        pn = R["panel"][r][key]
        ps = " / ".join(f"{pn[f'{main} squared h={h}']['p_holm']:.3f}" for h in HORIZONS if f"{main} squared h={h}" in pn)
        pa = " / ".join(f"{pn[f'{main} absolute h={h}']['p_holm']:.3f}" for h in HORIZONS if f"{main} absolute h={h}" in pn)
        L.append(f"| {key} | {w['better']}/{w['total']} | {w['median_dMSE_pct']:+.1f}% | {ps} | {pa} |")
    L.append("")
L.append("## Scaling curve (7 holdout stations: 3 AU + 4 ID)\n\n| k | draws | mean NSE | sd | MSE change vs L-G3 |\n|---|---|---|---|---|")
for _, row in T3.iterrows(): L.append(f"| {row['k_stations']} | {row['draws']} | {row['NSE_mean']:.3f} | {row['NSE_sd']} | {row['dMSE_vs_LG3_pct']:+.1f}% |")
open(os.path.join(REPORTS, "summary.md"), "w", newline="\n").write("\n".join(L) + "\n"); print("\n".join(L))
