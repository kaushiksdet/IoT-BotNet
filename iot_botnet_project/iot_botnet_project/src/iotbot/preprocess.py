import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from .common import *

def prepare(df, k=30, test_size=0.20, drop_cols=(), seed=SEED):
    """Split BEFORE fitting preprocessing/selection (no leakage). Returns a dict of everything needed later."""
    num = [c for c in NUMERIC if c not in drop_cols]; cat = [c for c in CATEGORICAL if c not in drop_cols]
    feats = num + cat
    Xr, yr = df[feats], df[TARGET]
    Xtr, Xte, ytr, yte, ctr, cte, str_, ste = train_test_split(
        Xr, yr, df[CATEGORY], df[SUBCATEGORY], test_size=test_size, random_state=seed, stratify=yr)
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=True))]), cat)],
        sparse_threshold=1.0)
    Ptr, Pte = pre.fit_transform(Xtr), pre.transform(Xte)
    kk = min(k, Ptr.shape[1]); sel = SelectKBest(f_classif, k=kk).fit(Ptr, ytr)
    names_all = pre.get_feature_names_out()
    scores = pd.DataFrame({"feature": names_all, "f_score": sel.scores_, "selected": sel.get_support()}
                          ).sort_values("f_score", ascending=False)
    return dict(Xtr=sel.transform(Ptr), Xte=sel.transform(Pte), ytr=ytr.values, yte=yte.values,
                cat_te=cte.values, sub_te=ste.values, names=names_all[sel.get_support()],
                scores=scores, pre=pre, sel=sel, raw_te=Xte, n_encoded=Ptr.shape[1])

def dense(X): return X.toarray() if hasattr(X, "toarray") else np.asarray(X)
def cond(a):  # signed-log conditioning for deep models
    return np.sign(a) * np.log1p(np.abs(a))
