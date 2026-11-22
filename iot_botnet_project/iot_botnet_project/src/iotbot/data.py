"""RAM-safe loading: stream a (possibly 1M-row) headerless CSV and keep a stratified sample."""
import numpy as np, pandas as pd
from .common import BOT_IOT_COLUMNS, CATEGORICAL, NUMERIC, TARGET, SEED

def _reader(path, chunk, **kw):
    return pd.read_csv(path, header=None, names=BOT_IOT_COLUMNS, chunksize=chunk, low_memory=False, **kw)

def load_sample(path, sample_size=200_000, chunk=50_000, seed=SEED, verbose=True):
    """Pass 1 counts rows per class; pass 2 keeps the same fraction of every class (stratified)."""
    total = 0; counts = {}
    for c in _reader(path, chunk, usecols=[TARGET]):
        total += len(c)
        for k, v in c[TARGET].value_counts().items(): counts[k] = counts.get(k, 0) + int(v)
    frac = min(1.0, sample_size / total)
    if verbose: print(f"Scanned {total:,} rows; class counts {counts}; sampling fraction {frac:.3f}")
    parts = []
    for i, c in enumerate(_reader(path, chunk)):
        parts.append(c.groupby(TARGET, group_keys=False).sample(frac=frac, random_state=seed + i))
    df = pd.concat(parts, ignore_index=True)
    return df, total, counts

def clean(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.drop_duplicates()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].astype("int8")
    for c in NUMERIC: df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in CATEGORICAL: df[c] = df[c].map(lambda v: np.nan if pd.isna(v) else str(v))
    return df.reset_index(drop=True)
