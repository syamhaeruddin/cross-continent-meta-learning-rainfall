# Cross-Continent Meta-Learning for Monthly Rainfall Forecast Combination

Is the combination rule learned by a cross-station meta-learner gate general across climates and data types? This experiment trains one gate jointly on two continents — 15 SILO rain-gauge stations in Australia (1926–2025) and 20 NASA POWER grid locations in Indonesia (1981–2025) — and tests it in three ways: joint training (does pooling across continents help each region?), leave-region-out transfer (does a gate trained on one continent beat the local gates of the other?), and a scaling curve (does holdout accuracy grow with the number of pooled stations?). Haeruddin, Purwanto, Muljono, Edi Noersasongko.

It builds directly on the two source studies and reuses their meta-datasets and stored gate predictions; **no base model is retrained**:

- Australia: [cross-station-meta-learning-monthly-rainfall](https://github.com/syamhaeruddin/cross-station-meta-learning-monthly-rainfall) (test 2015–2025, 13 evaluated stations)
- Indonesia: [cross-station-meta-learning-indonesia-rainfall](https://github.com/syamhaeruddin/cross-station-meta-learning-indonesia-rainfall) (test 2018–2025, 20 locations)

## Design

| | Setting |
|---|---|
| Inputs | Per meta-sample: 12-month scaled error history of the 11 members, member forecasts, sin–cos of the target month, regime features, one-hot horizon, target — as stored by the source repositories (`results/meta_dataset.npz`) |
| Harmonisation | 11-member pool aligned by position (9 base models + expanding climatology + windowed climatology: 30 years AU, 20 years ID); regime features reduced to the three indices available in both regions (Niño3.4, DMI, SOI; standardised with each region's pre-validation data); six static station features recomputed and standardised over all 35 stations; time splits remain those of each region |
| Gate | Identical to the source studies: LSTM-16 over the error history, dense-16, softmax weights, Adam 10⁻³, MSE on scaled targets, L2, early stopping (patience 30) on the region validation periods, 3 seeds averaged |
| **RG** | Region-only gate with the harmonised (3-index) features, station embedding + static features — the control that separates the pooling effect from the feature change |
| **JG-E / JG-S** | Joint gates trained on all 35 stations, with station embedding + static features (JG-E) or static features only (JG-S) |
| **LRO** | Leave-region-out: trained on one continent only (no embedding, NNLS anchor pooled over the training region), tested on the other |
| **Scaling curve** | 7 fixed holdout stations (Darwin, Brisbane, Woomera; Banda Aceh, Jakarta, Banjarmasin, Ambon); transfer gates trained on k = 4, 8, 14, 28 stations drawn at random from the other 28 (5 draws, 1 for k = 28) |
| Benchmarks | Stored local gates (L-G3, L-G4), global gates (GG3, GG4), transfer gate (GG4-T), NNLS, simple mean, windowed climatology of each source study |
| Tests | Per-station DM-HLN tests with Holm correction within region (52 / 80 combinations); panel DM tests per region with Holm over the four horizons, squared and absolute errors |
| Hypotheses (fixed before the run) | H1 JG-E ≥ RG in both regions; H2 LRO ≥ L-G3 of the target region; H3 LRO ≈ GG4-T of the target region (exploratory); H4 holdout NSE increases with k (exploratory) |

## Main results

Mean NSE across the evaluated stations of each region (test periods 2015–2025 and 2018–2025; gates averaged over three seeds).

| Horizon | Australia: L-G3 | GG3 | GG4-T | RG | JG-S | **JG-E** | LRO (ID→AU) | | Indonesia: L-G3 | GG3 | GG4-T | RG | JG-S | **JG-E** | LRO (AU→ID) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.326 | 0.340 | 0.338 | 0.339 | 0.338 | **0.340** | 0.340 | | 0.388 | 0.407 | 0.420 | 0.402 | 0.409 | **0.412** | 0.399 |
| 3 | 0.329 | 0.340 | 0.332 | 0.337 | 0.337 | **0.338** | 0.337 | | 0.350 | 0.368 | 0.380 | 0.360 | 0.369 | **0.374** | 0.358 |
| 6 | 0.323 | 0.333 | 0.329 | 0.330 | 0.328 | **0.329** | 0.323 | | 0.320 | 0.340 | 0.335 | 0.333 | 0.349 | **0.353** | 0.339 |
| 12 | 0.321 | 0.329 | 0.324 | 0.327 | 0.325 | **0.325** | 0.325 | | 0.320 | 0.342 | 0.343 | 0.334 | 0.342 | **0.346** | 0.330 |
| mean | 0.325 | 0.335 | 0.331 | 0.333 | 0.332 | **0.333** | 0.331 | | 0.344 | 0.364 | 0.369 | 0.358 | 0.367 | **0.371** | 0.357 |

Benchmarks: NNLS 0.325 (AU) / 0.330 (ID), simple mean 0.321 / 0.323, windowed climatology 0.300 / 0.314.

- **H1 — joint training (JG-E vs RG).** In Indonesia the joint gate is more accurate than the region-only gate of identical architecture and features in 53 of 80 location–horizon combinations, with panel Holm-adjusted *p* < 0.001, < 0.001, 0.010 at horizons 1, 3, 6 months (0.13 at 12 months), and the same holds with absolute errors (*p* ≤ 0.006 at all horizons). In Australia the two are indistinguishable (21 of 52 wins, *p* ≥ 0.64): pooling the Indonesian meta-samples neither helps nor hurts the Australian stations. H1 is supported in the direction "pooling does not hurt and helps the smaller region".
- **Feature change (RG vs GG3).** Reducing the regime features from six to three costs accuracy in Indonesia (RG 0.358 vs GG3 0.364; *p* ≤ 0.009 at all horizons) and nothing in Australia. The joint gate recovers that loss with pooled samples alone: JG-E (three indices) is at least as accurate as GG3 (six indices) in Indonesia (46 of 80 wins; *p* ≥ 0.37 squared, 0.018–0.082 absolute in favour of JG-E) and reaches the highest mean NSE of all in-sample gates (0.371).
- **H2 — leave-region-out transfer (LRO vs L-G3).** A gate that has never seen the target continent beats that continent's local gates: Indonesia (gate trained on Australia) 49 of 80 wins, *p* < 0.001 and 0.007 at horizons 1 and 3 months (0.056, 0.165 at 6 and 12); Australia (gate trained on Indonesia) 31 of 52 wins, *p* = 0.019 at horizon 1 (0.068 at 3). LRO also beats the simple mean and the climatology at most horizons in both regions, and NNLS in neither.
- **H3 — transfer across vs within continent (LRO vs GG4-T).** In Australia the two transfer gates are equivalent (*p* ≥ 0.85). In Indonesia the within-continent transfer gate GG4-T remains better at horizons 1 and 3 months (*p* < 0.001, 0.003) and 12 months (0.045): transfer from a different climate and data type is real but weaker than transfer from the same region.
- **H4 — scaling curve.** Holdout NSE rises monotonically with the number of pooled training stations: 0.339 ± 0.010 (k = 4), 0.348 ± 0.005 (8), 0.349 ± 0.003 (14), 0.350 (28), against 0.328 for the stored local gates of the same stations; the spread across draws shrinks with k. The gain saturates after about 8 stations, i.e. the benefit of pooling comes mainly from escaping the single-station sample size, not from ever more stations.
- **Weights.** The joint gate learns different compositions per continent from the same parameters: at horizon 1 it assigns 0.36 of the weight to the climatology members and 0.25 to SARIMA in Australia, 0.21 and 0.43 in Indonesia (Table 4).
- **Effect size.** Median MSE reductions are small (JG-E vs L-G3: +1.0% AU, +2.2% ID; vs NNLS: +1.2% AU, −0.1% ID), as in the source studies; the contribution is consistency across stations and transferability, not a large gain.

Full tables and figures: `reports/` (Tables 1–4 as CSV, Figures 1–4, `summary.md`); all comparisons, per-station tests, weights, and training epochs: `results/evaluation_results.json`.

## Repository layout

```
├── notebooks/Cross_Continent_Gate_Experiment.ipynb   # runs the whole experiment and shows the tables and figures
├── src/
│   ├── common.py          # region settings, pool, DM-HLN test, Holm correction
│   ├── fetch_inputs.py    # stage 0: copy meta-datasets and stored predictions from the two source repositories
│   ├── harmonise.py       # stage 1: joint meta-dataset (data/joint/meta_joint.npz)
│   ├── gates.py           # stage 2: RG, JG-E, JG-S, LRO, scaling-curve gates
│   ├── run_all.py         # runs all gate trainings in parallel
│   ├── evaluate.py        # stage 3: NSE, DM tests (Holm), panel tests, weights, scaling curve
│   └── report.py          # stage 4: tables and figures
├── data/joint/            # harmonised joint meta-dataset (committed)
├── data/input/            # copies of the source repositories' outputs (not committed; rebuilt by fetch_inputs.py)
├── results/               # pred_*.npz of every gate, evaluation_results.json
├── reports/               # Tables 1-4 (CSV), Figures 1-4 (PNG), summary.md
├── requirements.txt
└── README.md
```

## How to run

Python 3.13 is recommended. Clone the two source repositories as sibling folders of this one (or pass their paths to `fetch_inputs.py`).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/Cross_Continent_Gate_Experiment.ipynb
```

The notebook has one switch, `RETRAIN_GATES`: `False` (default) re-evaluates the stored gate outputs in about a minute; `True` retrains all 22 gate configurations × 3 seeds (about 5 minutes with 8 parallel processes on a 16-core CPU). Command-line equivalent:

```bash
python src/fetch_inputs.py [path_to_australia_repo path_to_indonesia_repo]
python src/harmonise.py
python src/run_all.py all        # gates (existing files are skipped) + evaluate + report
```

Parallel processes: environment variable `WORKERS` (default 6); gate seeds: `SEEDS` (default `1,2,3`).

## Anti-leakage notes

Every input to a meta-sample precedes the forecast origin, as in the source studies. In addition: the static-feature standardisation uses only pre-validation data of each region; the NNLS anchor of the leave-region-out and scaling-curve gates is fitted on the training stations only; the holdout stations of the scaling curve never enter training or validation; no hyperparameter is tuned on any test period, and each region's test period is touched once per configuration.

## Contact

Haeruddin, Faculty of Computer Science, Universitas Dian Nuswantoro, Semarang, and Universitas Internasional Batam — haeruddin@uib.ac.id
