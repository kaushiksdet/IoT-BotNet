"""Review 2: MLP / 1-D CNN / LSTM / Autoencoder, validation-only thresholds, full metrics, per-attack analysis,
permutation importance, ports+timestamps ablation, latency. Needs outputs/ from 01_review1_classical.py."""
import argparse, os, sys, time, joblib, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from iotbot.common import *; from iotbot.preprocess import prepare, dense, cond
from iotbot.metrics import full_metrics, balanced_threshold, security_threshold
from iotbot import deep, plots

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="outputs"); ap.add_argument("--quick", action="store_true")
ap.add_argument("--sec-recall", type=float, default=0.999); a = ap.parse_args()
EP = {"MLP": 30, "CNN": 25, "LSTM": 12, "AE": 60} if not a.quick else {k: 2 for k in ["MLP", "CNN", "LSTM", "AE"]}
set_seed(); D = dirs(a.out)
P = joblib.load(os.path.join(D["cache"], "split_main.joblib")); CL = joblib.load(os.path.join(D["models"], "classical_models.joblib"))
Xtr, Xte, ytr, yte = dense(P["Xtr"]), dense(P["Xte"]), P["ytr"], P["yte"]; Ctr, Cte = cond(Xtr), cond(Xte)
Xa, Xv, ya, yv = deep.val_split(Ctr, ytr)
print(f"Train {Xa.shape} | val {Xv.shape} | test {Xte.shape} | class weights {deep.sqrt_class_weights(ya)}")
SC, PRED, ROWS, HIST, TRAIN_S, THR, MODELS = {}, {}, [], {}, {}, {}, {}

for nm, m in CL.items():   # classical baselines at their default rule
    s = m.decision_function(P["Xte"]) if nm == "SVM" else m.predict_proba(P["Xte"])[:, 1]
    pred = (s > 0).astype(int) if nm == "SVM" else m.predict(P["Xte"]); SC[nm], PRED[nm] = s, pred
    r = full_metrics(yte, pred, s, nm); r["operating_point"] = "default"; ROWS.append(r)

for nm, b, ep in [("MLP", deep.build_mlp, EP["MLP"]), ("CNN", deep.build_cnn, EP["CNN"]), ("LSTM", deep.build_lstm, EP["LSTM"])]:
    m, h, secs = deep.train_classifier(b, Xa, ya, Xv, yv, ep); print(f"{nm}: {len(h['loss'])} epochs, {secs:.0f}s")
    sv, st = m.predict(Xv, batch_size=8192, verbose=0).ravel(), m.predict(Cte, batch_size=8192, verbose=0).ravel()
    tb, ts = balanced_threshold(yv, sv), security_threshold(yv, sv, a.sec_recall); THR[nm] = dict(default=0.5, balanced=tb, security_first=ts)
    SC[nm], PRED[nm], HIST[nm], TRAIN_S[nm], MODELS[nm] = st, (st >= tb).astype(int), h, secs, m
    for op, t in [("default", .5), ("balanced", tb), ("security_first", ts)]:
        r = full_metrics(yte, (st >= t).astype(int), st, nm); r["operating_point"] = op; r["threshold"] = t; r["train_s"] = secs; ROWS.append(r)
    m.save(os.path.join(D["models"], f"{nm.lower()}.keras"))

# autoencoder: trained on NORMAL flows only; score = reconstruction error
ae, h, secs = deep.train_autoencoder(Ctr[ytr == 0], EP["AE"]); HIST["AE"] = h; TRAIN_S["Autoencoder"] = secs
sv, st = deep.recon_error(ae, Xv), deep.recon_error(ae, Cte)
tb, ts = balanced_threshold(yv, sv), security_threshold(yv, sv, a.sec_recall); THR["Autoencoder"] = dict(balanced=tb, security_first=ts)
SC["Autoencoder"], PRED["Autoencoder"] = st, (st >= tb).astype(int)
for op, t in [("balanced", tb), ("security_first", ts)]:
    r = full_metrics(yte, (st >= t).astype(int), st, "Autoencoder"); r["operating_point"] = op; r["threshold"] = t; r["train_s"] = secs; ROWS.append(r)
ae.save(os.path.join(D["models"], "autoencoder.keras"))

R = pd.DataFrame(ROWS); R.to_csv(os.path.join(D["results"], "review2_all_operating_points.csv"), index=False)
main = R[R.operating_point.isin(["default", "balanced"])].copy()
main = main[(main.model.isin(CL)) & (main.operating_point == "default") | (~main.model.isin(CL)) & (main.operating_point == "balanced")]
main.to_csv(os.path.join(D["results"], "review2_comparison.csv"), index=False); save_json(THR, os.path.join(D["results"], "thresholds.json"))
print(main[["model", "fp", "fn", "recall", "specificity", "mcc", "roc_auc", "pr_auc_normal"]].round(4).to_string(index=False))

COL = {m: ("#2a78d6" if m in CL else "#eb6834") for m in main.model}
order = list(main.sort_values("mcc").model)
plots.bars(order, [float(main.set_index("model").loc[m, "mcc"]) for m in order], [COL[m] for m in order], "MCC (blue classical, orange deep)", os.path.join(D["figures"], "09_mcc_comparison.png"))
plots.roc_pr(yte, SC, COL, os.path.join(D["figures"], "10_roc_pr.png"))
plots.confusion_grid({r.model: r._asdict() for r in main.itertuples()}, COL, os.path.join(D["figures"], "11_confusion_all.png"))
plots.curves({k: v for k, v in HIST.items()}, os.path.join(D["figures"], "12_training_curves.png"))

# per-attack-category detection rate (recall) at the reported operating point
cats = ["Reconnaissance", "DoS", "Theft"]; cat = P["cat_te"]; ms = [m for m in PRED if m != "Autoencoder"]
M = np.array([[PRED[m][cat == c].mean() * 100 if (cat == c).any() else np.nan for c in cats] for m in ms])
pd.DataFrame(M, index=ms, columns=cats).round(2).to_csv(os.path.join(D["results"], "per_attack_detection.csv"))
plots.heatmap(np.nan_to_num(M), ms, [f"{c}\n(n={(cat==c).sum()})" for c in cats], "Detection rate by attack category", os.path.join(D["figures"], "13_per_attack.png"))

# permutation importance (MLP, validation ROC-AUC drop)
rng = np.random.default_rng(SEED); base = roc_auc_score(yv, MODELS["MLP"].predict(Xv, batch_size=8192, verbose=0).ravel()); imp = []
for j in range(Xv.shape[1]):
    Xp = Xv.copy(); Xp[:, j] = rng.permutation(Xp[:, j]); imp.append(base - roc_auc_score(yv, MODELS["MLP"].predict(Xp, batch_size=8192, verbose=0).ravel()))
I = pd.DataFrame({"feature": P["names"], "auc_drop": imp}).sort_values("auc_drop", ascending=False); I.to_csv(os.path.join(D["results"], "permutation_importance.csv"), index=False)
top = I.head(12)[::-1]; plots.bars(list(top.feature), list(top.auc_drop), "#eb6834", "MLP permutation importance (ROC-AUC drop)", os.path.join(D["figures"], "14_permutation_importance.png"))

# ablation: remove ports + timestamps (testbed-specific shortcuts), retrain RF and MLP
df = pd.read_pickle(os.path.join(D["cache"], "sample.pkl")); P2 = prepare(df, drop_cols=SHORTCUT_COLS)
X2tr, X2te = dense(P2["Xtr"]), dense(P2["Xte"]); C2a, C2v, y2a, y2v = deep.val_split(cond(X2tr), P2["ytr"])
rf2 = RandomForestClassifier(200, n_jobs=-1, random_state=SEED).fit(P2["Xtr"], P2["ytr"]); r_rf = full_metrics(P2["yte"], rf2.predict(P2["Xte"]), rf2.predict_proba(P2["Xte"])[:, 1], "Random Forest")
m2, _, _ = deep.train_classifier(deep.build_mlp, C2a, y2a, C2v, y2v, EP["MLP"]); t2 = balanced_threshold(y2v, m2.predict(C2v, verbose=0).ravel())
s2 = m2.predict(cond(X2te), verbose=0).ravel(); r_ml = full_metrics(P2["yte"], (s2 >= t2).astype(int), s2, "MLP")
mm = main.set_index("model"); keys = ["specificity", "recall", "mcc", "fp", "fn"]
AB = pd.DataFrame([dict(model=m, features="with ports+time", **{k: mm.loc[m, k] for k in keys}) for m in ["Random Forest", "MLP"]] +
                  [dict(model=r["model"], features="without ports+time", **{k: r[k] for k in keys}) for r in (r_rf, r_ml)])
AB.to_csv(os.path.join(D["results"], "ablation_no_ports_time.csv"), index=False); print(AB.round(4).to_string(index=False))

# latency (model only, ms per flow, single-flow calls)
lat = {}; X1 = Xte[:100]
for nm, m in CL.items():
    Xs = P["Xte"][:100]; t = time.time(); [m.predict(Xs[i]) for i in range(100)]; lat[nm] = (time.time() - t) * 10
for nm, m in MODELS.items():
    t = time.time(); [m(cond(X1[i:i + 1]).astype("float32"), training=False) for i in range(100)]; lat[nm] = (time.time() - t) * 10
t = time.time(); [ae(cond(X1[i:i + 1]).astype("float32"), training=False) for i in range(100)]; lat["Autoencoder"] = (time.time() - t) * 10
pd.DataFrame({"model": list(lat), "ms_per_flow": list(lat.values()), "train_s": [TRAIN_S.get(k, np.nan) for k in lat]}).to_csv(os.path.join(D["results"], "latency_cost.csv"), index=False)
joblib.dump(dict(thresholds=THR), os.path.join(D["cache"], "deep_meta.joblib")); print("Review 2 done. Results in", D["results"])
