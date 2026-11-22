"""Create a synthetic Bot-IoT-style CSV (same 35 columns, no header) so the project runs without the real dataset."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from iotbot.synth import make_synthetic
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=60000); ap.add_argument("--out", default="data/synthetic_botiot.csv")
a = ap.parse_args(); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
df = make_synthetic(a.n); df.to_csv(a.out, header=False, index=False)
print(f"Wrote {a.out}: {len(df):,} rows\n{df.category.value_counts()}")
