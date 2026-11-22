"""Near-real-time demo: replay raw flows in micro-batches -> preprocess -> MLP score -> ALERT."""
import argparse, json, os, sys, time, joblib, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tensorflow import keras
from iotbot.common import *; from iotbot.preprocess import dense, cond
ap = argparse.ArgumentParser(); ap.add_argument("--out", default="outputs"); ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--batch", type=int, default=10); ap.add_argument("--rate", type=float, default=0, help="flows/s pacing; 0 = as fast as possible")
ap.add_argument("--operating-point", default="security_first", choices=["balanced", "security_first"]); a = ap.parse_args()
set_seed(); D = dirs(a.out); P = joblib.load(os.path.join(D["cache"], "split_main.joblib"))
thr = json.load(open(os.path.join(D["results"], "thresholds.json")))["MLP"][a.operating_point]
mlp = keras.models.load_model(os.path.join(D["models"], "mlp.keras")); raw = P["raw_te"].copy(); y = P["yte"]
rng = np.random.default_rng(SEED); normal = np.where(y == 0)[0]; att = np.where(y == 1)[0]
idx = rng.permutation(np.r_[normal[:min(len(normal), a.n // 4)], rng.choice(att, a.n - min(len(normal), a.n // 4), replace=False)])
lat, tp, fp, fn, tn = [], 0, 0, 0, 0
for s in range(0, len(idx), a.batch):
    b = idx[s:s + a.batch]; t0 = time.time()
    Z = cond(dense(P["sel"].transform(P["pre"].transform(raw.iloc[b])))); p = mlp.predict(Z, verbose=0).ravel(); al = p >= thr
    dt = (time.time() - t0) * 1000 / len(b); lat += [dt] * len(b)
    for k, (alert, truth) in enumerate(zip(al, y[b])):
        tp += alert and truth == 1; fp += alert and truth == 0; fn += (not alert) and truth == 1; tn += (not alert) and truth == 0
    if s // a.batch % 20 == 0: print(f"flows {s + len(b):5d} | alerts so far {tp + fp:5d} | FP {fp} | FN {fn} | {dt:.1f} ms/flow")
    if a.rate: time.sleep(max(0, len(b) / a.rate - (time.time() - t0)))
print(f"\nReplayed {len(idx):,} flows | alerts {tp + fp:,} (true {tp:,}, false {fp}) | missed {fn} | normals passed {tn}")
print(f"Latency per flow (end to end): median {np.median(lat):.1f} ms, p95 {np.percentile(lat, 95):.1f} ms | ~{1000 / np.mean(lat):.0f} flows/s")
