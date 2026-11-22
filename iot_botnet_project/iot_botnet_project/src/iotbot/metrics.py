import numpy as np
from scipy.stats import beta
from sklearn.metrics import (confusion_matrix, matthews_corrcoef, f1_score, roc_curve, roc_auc_score,
                             average_precision_score)

def cp_ci(k, n, a=0.05):
    if n == 0: return (np.nan, np.nan)
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return float(lo), float(hi)

def full_metrics(y, pred, score=None, name=""):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    P, N = tp + fn, tn + fp
    m = dict(model=name, tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
             precision=tp / max(tp + fp, 1), recall=tp / max(P, 1), specificity=tn / max(N, 1),
             f1=f1_score(y, pred, zero_division=0), mcc=matthews_corrcoef(y, pred),
             fpr=fp / max(N, 1), fnr=fn / max(P, 1))
    m["fpr_ci_lo"], m["fpr_ci_hi"] = cp_ci(fp, N); m["fnr_ci_lo"], m["fnr_ci_hi"] = cp_ci(fn, P)
    if score is not None:
        m["roc_auc"] = roc_auc_score(y, score); m["pr_auc_attack"] = average_precision_score(y, score)
        m["pr_auc_normal"] = average_precision_score(1 - y, -score)
    return m

def balanced_threshold(y, s):            # Youden J = max(TPR - FPR); validation data only
    fpr, tpr, t = roc_curve(y, s); return float(t[np.argmax(tpr - fpr)])

def security_threshold(y, s, min_recall=0.999):   # lowest FPR with recall >= min_recall
    fpr, tpr, t = roc_curve(y, s); ok = np.where(tpr >= min_recall)[0]
    return float(t[ok[0]]) if len(ok) else balanced_threshold(y, s)
