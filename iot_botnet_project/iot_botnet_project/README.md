# IoT Botnet Attack Detection (UNSW Bot-IoT): classical ML + deep learning

Plain-Python version of the Review 1 + Review 2 experiment (no Colab needed).

| Stage | What it does |
|---|---|
| `01_review1_classical.py` | Chunked scan of the 1M-row CSV, stratified 200k sample, cleaning, 80/20 split **before** preprocessing, sparse one-hot (~84k columns), ANOVA top-30, Decision Tree / Random Forest / SVM (RBF on 50k rows) |
| `02_review2_deep.py` | MLP, 1-D CNN, LSTM, Autoencoder (normal flows only); signed-log inputs; sqrt-balanced class weights; thresholds chosen on validation only (default, balanced = Youden J, security-first = lowest FPR at recall >= 99.9%); precision/recall/specificity/F1/MCC, ROC-AUC, PR-AUC (attack and Normal), 95% Clopper-Pearson CIs; per-attack detection; permutation importance; ports + timestamps ablation; latency |
| `03_stream_demo.py` | Near-real-time demo: raw flows in micro-batches -> preprocessing -> MLP -> ALERT, with latency and throughput |
| `00_make_synthetic.py` | Optional: writes a synthetic Bot-IoT-style CSV (same 35 columns) so you can run everything without the real dataset |

## 1. Setup (Python 3.10 to 3.12 recommended)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```
No GPU is needed. If TensorFlow will not install, use `pip install tensorflow-cpu`.

## 2. Data

Put the dataset at `data/UNSW_2018_IoT_Botnet_Dataset_2.csv` (headerless, 35 columns, as downloaded from UNSW).
No dataset? Create a synthetic one: `python scripts/00_make_synthetic.py` (writes `data/synthetic_botiot.csv`), then pass `--data data/synthetic_botiot.csv`.

## 3. Run

```bash
# Everything, quick dry run (a few minutes, tiny sample and epochs):
python scripts/run_all.py --quick

# Full experiment (as in the Review 2 results):
python scripts/run_all.py

# Or step by step:
python scripts/01_review1_classical.py --data data/UNSW_2018_IoT_Botnet_Dataset_2.csv
python scripts/02_review2_deep.py
python scripts/03_stream_demo.py --n 2000 --batch 10
```
Useful options: `--sample-size 200000`, `--svm-sample 50000`, `--out outputs`, `--sec-recall 0.999`,
`--operating-point balanced|security_first` (demo). Run 02 only after 01 (it reuses the same split and 30 features so the comparison is fair).

## 4. Outputs (in `outputs/`)

- `results/`: `review1_results.csv`, `review2_comparison.csv`, `review2_all_operating_points.csv`, `thresholds.json`, `per_attack_detection.csv`, `permutation_importance.csv`, `ablation_no_ports_time.csv`, `latency_cost.csv`, `feature_selection_scores.csv`
- `figures/`: class distribution, top features, confusion matrices, MCC comparison, ROC/PR, training curves, per-attack heatmap, permutation importance
- `models/`: classical models (joblib), MLP/CNN/LSTM/Autoencoder (`.keras`)
- `cache/`: the working sample and the fitted preprocessing + split (needed by 02 and 03)

## 5. Reproducibility and notes

- Fixed seed 42 everywhere; the test set is never used for tuning or thresholds.
- Time on a laptop CPU (full run): scan about a few minutes, classical about 5-10 min (SVM on 50k rows is the slowest), deep models about 10-20 min (the LSTM is the slowest).
- Memory: the CSV is streamed in 50k-row chunks; the encoded matrix is sparse.
- The test split has only a few hundred normal flows, so FPR confidence intervals are wide. Always read FP/FN counts alongside them.
- Findings from the reference run: Random Forest was best overall; the MLP was the best deep model; the autoencoder was weaker than the supervised models; removing ports and timestamps hurt the MLP more than the Random Forest. Your numbers may differ slightly across library versions.

## Layout
```
iot_botnet_project/
  requirements.txt  README.md
  data/             <- put the CSV here
  scripts/          00_make_synthetic.py  01_review1_classical.py  02_review2_deep.py  03_stream_demo.py  run_all.py
  src/iotbot/       common.py data.py preprocess.py metrics.py deep.py plots.py synth.py
  outputs/          <- created by the scripts
```
