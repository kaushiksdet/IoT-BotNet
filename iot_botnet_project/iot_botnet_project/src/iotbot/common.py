import os, json, random
import numpy as np

SEED = 42

BOT_IOT_COLUMNS = [
    "pkSeqID", "stime", "flgs", "proto", "saddr", "sport", "daddr", "dport",
    "pkts", "bytes", "state", "ltime", "seq", "dur", "mean", "stddev",
    "smac", "dmac", "sum", "min", "max", "soui", "doui", "sco", "dco",
    "spkts", "dpkts", "sbytes", "dbytes", "rate", "srate", "drate",
    "attack", "category", "subcategory",
]
TARGET, CATEGORY, SUBCATEGORY = "attack", "category", "subcategory"

# identifiers and labels never used as model inputs
DROP_FROM_FEATURES = [TARGET, CATEGORY, SUBCATEGORY, "pkSeqID", "seq", "saddr", "daddr",
                      "smac", "dmac", "soui", "doui", "sco", "dco"]
CATEGORICAL = ["flgs", "proto", "sport", "dport", "state"]
NUMERIC = ["stime", "pkts", "bytes", "ltime", "dur", "mean", "stddev", "sum", "min", "max",
           "spkts", "dpkts", "sbytes", "dbytes", "rate", "srate", "drate"]
# removed in the robustness ablation (testbed-specific shortcuts)
SHORTCUT_COLS = ["sport", "dport", "stime", "ltime"]

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    try:
        import tensorflow as tf; tf.random.set_seed(seed)
    except Exception:
        pass

def dirs(out):
    d = {k: os.path.join(out, k) for k in ["figures", "results", "models", "cache"]}
    for p in d.values(): os.makedirs(p, exist_ok=True)
    return d

def save_json(obj, path):
    with open(path, "w") as f: json.dump(obj, f, indent=2, default=float)
