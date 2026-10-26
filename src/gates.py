# Stage 2: meta-learner gates trained on the harmonised joint meta-dataset. Architecture and hyperparameters are those of the
# source studies (LSTM-16 over the error history, softmax weights, Adam 1e-3, MSE on scaled targets, L2, early stopping, 3 seeds).
# Usage:
#   python src/gates.py region <australia|indonesia>   -> RG: region-only gate with embedding + static features (control for the
#                                                         change from 6 to 3 regime features)            -> results/pred_rg_<region>.npz
#   python src/gates.py joint <JG-E|JG-S>              -> joint gate on all 35 stations; JG-E with station embedding + static features,
#                                                         JG-S with static features only                   -> results/pred_jg_<name>.npz
#   python src/gates.py lro <australia|indonesia>      -> leave-region-out: trained on the named region only (no embedding, NNLS anchor
#                                                         pooled over the training region), tested on the other region -> results/pred_lro_<train region>.npz
#   python src/gates.py scale <k> <draw>               -> scaling curve: trained on k stations drawn at random (seed = draw) from the 28
#                                                         non-holdout stations, tested on the 7 holdout stations -> results/pred_scale_k<k>_d<draw>.npz
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from scipy.optimize import nnls

SEEDS = [int(s) for s in os.environ.get("SEEDS", "1,2,3").split(",")]
SCALE_K = [4, 8, 14, 28]; SCALE_DRAWS = 5

import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks, regularizers
tf.config.threading.set_intra_op_parallelism_threads(1); tf.config.threading.set_inter_op_parallelism_threads(1)
try: tf.config.experimental.enable_op_determinism()
except AttributeError: pass


def seed(s): np.random.seed(s); tf.keras.utils.set_random_seed(s)
def prior(w):
    w = np.maximum(w / max(w.sum(), 1e-9), 0.01); return np.log(w / w.sum())


def build_gate(M, n_st, n_rg, anchor, use_emb, use_static, n_static):
    ei = layers.Input((12, M), name="E"); fi = layers.Input((M,), name="P"); mo = layers.Input((2,), name="Mo")
    rg = layers.Input((n_rg,), name="Rg"); hh = layers.Input((4,), name="H"); ins = [ei, fi, mo, rg, hh]
    parts = [layers.LSTM(16, kernel_regularizer=regularizers.l2(1e-3))(ei), fi, mo, rg, hh]
    if use_emb:
        si = layers.Input((1,), name="st", dtype="int32"); ins.append(si)
        parts.append(layers.Flatten()(layers.Embedding(n_st, 4, embeddings_regularizer=regularizers.l2(1e-3))(si)))
    if use_static:
        sf = layers.Input((n_static,), name="sf"); ins.append(sf); parts.append(sf)
    z = layers.Dropout(0.2)(layers.Concatenate()(parts)); z = layers.Dense(16, activation="relu", kernel_regularizer=regularizers.l2(1e-3))(z)
    if anchor:
        lp = layers.Input((M,), name="LP"); ins.append(lp)
        lg = layers.Add()([layers.Dense(M, kernel_regularizer=regularizers.l2(1e-2), kernel_initializer="zeros")(z), lp])
    else:
        lg = layers.Dense(M)(z)
    w = layers.Softmax(name="w")(lg); m = Model(ins, layers.Dot(axes=1)([w, fi]))
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse"); return m


def inputs(D, k, anchor, use_emb, use_static, LP=None):
    X = [D["E"][k], D["P"][k], D["Mo"][k], D["Rg"][k], D["H"][k]]
    if use_emb: X.append(D["st"][k][:, None])
    if use_static: X.append(D["static"][D["st"][k]])
    if anchor: X.append(LP[k])
    return X


def fit_predict(D, tr, va, te, anchor, use_emb, use_static, LP, batch, tag):
    M = D["P"].shape[1]; preds, wts, epochs = [], [], []
    for s_ in SEEDS:
        seed(s_); m = build_gate(M, len(D["stations"]), D["Rg"].shape[1], anchor, use_emb, use_static, D["static"].shape[1])
        hist = m.fit(inputs(D, tr, anchor, use_emb, use_static, LP), D["y"][tr], epochs=300, batch_size=batch, verbose=0,
                     validation_data=(inputs(D, va, anchor, use_emb, use_static, LP), D["y"][va]),
                     callbacks=[callbacks.EarlyStopping(patience=30, restore_best_weights=True)])
        wm = Model(m.inputs, m.get_layer("w").output)
        preds.append(m.predict(inputs(D, te, anchor, use_emb, use_static, LP), verbose=0, batch_size=2048).ravel())
        wts.append(wm.predict(inputs(D, te, anchor, use_emb, use_static, LP), verbose=0, batch_size=2048))
        epochs.append(len(hist.history["loss"])); print(tag, "seed", s_, "done, epochs", epochs[-1], flush=True)
    return np.array(preds), np.array(wts), np.array(epochs)


def pooled_prior(D, rows):
    """NNLS anchor pooled over the given training rows (pre-test, valid), one weight vector per horizon."""
    LP = np.zeros_like(D["P"])
    for j in range(4):
        kt = rows & (D["j"] == j) & (D["split"] < 2) & D["ok"]
        w, _ = nnls(D["P"][kt], D["y"][kt]); LP[D["j"] == j] = prior(w)
    return LP


def holdout_stations(D):
    return [D["stations"].index(s) for r in REGION_LIST for s in REGIONS[r]["holdout"]]


if __name__ == "__main__":
    cmd = sys.argv[1]; t0 = time.time(); D = load_joint()
    base_tr = (D["split"] == 0) & D["ok"]; base_va = (D["split"] == 1) & D["ok"]; is_te = D["split"] == 2
    if cmd == "region":
        r = sys.argv[2]; inr = D["region"] == REGION_LIST.index(r); te = np.where(inr & is_te)[0]
        p, w, ep = fit_predict(D, np.where(inr & base_tr)[0], np.where(inr & base_va)[0], te, False, True, True, None, 256, f"RG {r}")
        np.savez(os.path.join(OUT, f"pred_rg_{r}.npz"), idx=te, pred=p, w=w.mean(0), epochs=ep)
    elif cmd == "joint":
        name = sys.argv[2]; emb = name == "JG-E"; te = np.where(is_te)[0]
        p, w, ep = fit_predict(D, np.where(base_tr)[0], np.where(base_va)[0], te, False, emb, True, None, 256, name)
        np.savez(os.path.join(OUT, f"pred_jg_{name}.npz"), idx=te, pred=p, w=w.mean(0), epochs=ep)
    elif cmd == "lro":
        r = sys.argv[2]; inr = D["region"] == REGION_LIST.index(r); te = np.where((~inr) & is_te)[0]
        LP = pooled_prior(D, inr)   # anchor from the training region only
        p, w, ep = fit_predict(D, np.where(inr & base_tr)[0], np.where(inr & base_va)[0], te, True, False, True, LP, 256, f"LRO train={r}")
        np.savez(os.path.join(OUT, f"pred_lro_{r}.npz"), idx=te, pred=p, w=w.mean(0), epochs=ep)
    elif cmd == "scale":
        k, draw = int(sys.argv[2]), int(sys.argv[3]); hold = holdout_stations(D)
        cand = [i for i in range(len(D["stations"])) if i not in hold]
        rng = np.random.RandomState(100 + draw); chosen = sorted(rng.choice(cand, k, replace=False).tolist()) if k < len(cand) else cand
        intr = np.isin(D["st"], chosen); te = np.where(np.isin(D["st"], hold) & is_te)[0]
        LP = pooled_prior(D, intr)
        p, w, ep = fit_predict(D, np.where(intr & base_tr)[0], np.where(intr & base_va)[0], te, True, False, True, LP, 256, f"scale k={k} draw={draw}")
        np.savez(os.path.join(OUT, f"pred_scale_k{k}_d{draw}.npz"), idx=te, pred=p, w=w.mean(0), epochs=ep, train_stations=np.array(chosen))
    print(cmd, sys.argv[2:], "done", round((time.time() - t0) / 60, 1), "min", flush=True)
