"""Review 1: chunked scan -> stratified sample -> clean -> split -> sparse preprocessing -> ANOVA top-30 -> DT / RF / SVM."""
import argparse, os, sys, time, joblib, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from iotbot.common import *; from iotbot.data import load_sample, clean
from iotbot.preprocess import prepare; from iotbot.metrics import full_metrics; from iotbot import plots

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/UNSW_2018_IoT_Botnet_Dataset_2.csv"); ap.add_argument("--out", default="outputs")
ap.add_argument("--sample-size", type=int, default=200_000); ap.add_argument("--svm-sample", type=int, default=50_000)
ap.add_argument("--quick", action="store_true", help="small sample for a fast dry run")
a = ap.parse_args()
if a.quick: a.sample_size, a.svm_sample = 20_000, 5_000
set_seed(); D = dirs(a.out)

df, total, counts = load_sample(a.data, a.sample_size)
df = clean(df); print("Clean working sample:", df.shape)
df.to_pickle(os.path.join(D["cache"], "sample.pkl"))
plots.class_distribution({"Normal (0)": counts.get(0, 0), "Attack (1)": counts.get(1, 0)}, os.path.join(D["figures"], "01_normal_vs_attack.png"))
cc = df[CATEGORY].value_counts().to_dict(); plots.class_distribution(cc, os.path.join(D["figures"], "02_category_distribution.png"), "Records by category (working sample)")

P = prepare(df); print(f"Train {P['Xtr'].shape} | test {P['Xte'].shape} | normal in test: {(P['yte']==0).sum()} | encoded cols: {P['n_encoded']:,}")
P["scores"].to_csv(os.path.join(D["results"], "feature_selection_scores.csv"), index=False)
top = P["scores"].head(15)[::-1]; plots.bars(list(top.feature), list(top.f_score), "#2a78d6", "Top ANOVA F-scores", os.path.join(D["figures"], "03_top_features.png"))

rng = np.random.default_rng(SEED); idx = rng.choice(P["Xtr"].shape[0], min(a.svm_sample, P["Xtr"].shape[0]), replace=False)
models = {"Decision Tree": DecisionTreeClassifier(random_state=SEED), "Random Forest": RandomForestClassifier(200, n_jobs=-1, random_state=SEED),
          "SVM": SVC(kernel="rbf", random_state=SEED)}
rows, cms = [], {}
for nm, m in models.items():
    t = time.time()
    (m.fit(P["Xtr"][idx], P["ytr"][idx]) if nm == "SVM" else m.fit(P["Xtr"], P["ytr"])); secs = time.time() - t
    if nm == "SVM": s = m.decision_function(P["Xte"]); pred = (s > 0).astype(int)
    else: s = m.predict_proba(P["Xte"])[:, 1]; pred = m.predict(P["Xte"])
    r = full_metrics(P["yte"], pred, s, nm); r["train_s"] = secs; rows.append(r); cms[nm] = r
    print(f"{nm}: FP {r['fp']} FN {r['fn']} MCC {r['mcc']:.4f} ({secs:.0f}s)")
res = pd.DataFrame(rows); res.to_csv(os.path.join(D["results"], "review1_results.csv"), index=False)
plots.confusion_grid(cms, {k: "#2a78d6" for k in cms}, os.path.join(D["figures"], "04_review1_confusion.png"))
joblib.dump(models, os.path.join(D["models"], "classical_models.joblib")); joblib.dump(P, os.path.join(D["cache"], "split_main.joblib"))
print("Review 1 done. Results in", D["results"])
