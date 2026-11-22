import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
BLUE, ORG = "#2a78d6", "#eb6834"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})

def save(fig, path): fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)

def class_distribution(counts, path, title="Normal vs attack records"):
    fig, ax = plt.subplots(figsize=(6, 4)); ks = list(counts); v = [counts[k] for k in ks]
    ax.bar([str(k) for k in ks], v, color=[BLUE, ORG][:len(ks)] + [BLUE] * 8); ax.set_yscale("log"); ax.set_title(title)
    for i, x in enumerate(v): ax.text(i, x * 1.1, f"{x:,}", ha="center")
    save(fig, path)

def confusion_grid(cms, colors, path):
    n = len(cms); cols = min(4, n); rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows)); axs = np.atleast_1d(axs).ravel()
    for ax, (nm, cm) in zip(axs, cms.items()):
        M = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]); ax.imshow(np.log10(M + 1), cmap="Blues")
        for (i, j), v in np.ndenumerate(M): ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=9)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Normal", "Attack"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["Normal", "Attack"])
        ax.set_title(nm, color=colors[nm]); ax.set_xlabel("predicted"); ax.set_ylabel("actual")
    for ax in axs[n:]: ax.axis("off")
    save(fig, path)

def roc_pr(y, scores, colors, path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.5))
    for nm, s in scores.items():
        f, t, _ = roc_curve(y, s); a.plot(f, t, color=colors[nm], label=nm)
        p, r, _ = precision_recall_curve(1 - y, -s); b.plot(r, p, color=colors[nm], label=nm)
    a.plot([0, 1], [0, 1], color="#ccc"); a.set(title="ROC (attack = positive)", xlabel="FPR", ylabel="TPR")
    b.set(title="PR (Normal = positive)", xlabel="recall", ylabel="precision"); a.legend(fontsize=8); save(fig, path)

def bars(names, vals, colors, title, path, xlabel=""):
    fig, ax = plt.subplots(figsize=(6.5, 0.5 * len(names) + 1.5)); ax.barh(names, vals, color=colors); ax.set_title(title); ax.set_xlabel(xlabel)
    for i, v in enumerate(vals): ax.text(v, i, f" {v:.4g}", va="center", fontsize=8)
    save(fig, path)

def heatmap(M, rows, cols, title, path, fmt="{:.1f}%"):
    fig, ax = plt.subplots(figsize=(1.6 * len(cols) + 3, 0.5 * len(rows) + 2)); ax.imshow(M, cmap="Blues", vmin=0, vmax=100)
    for (i, j), v in np.ndenumerate(M): ax.text(j, i, fmt.format(v), ha="center", va="center", color="white" if v > 55 else "black")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols); ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows); ax.set_title(title); save(fig, path)

def curves(hist, path):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for nm, h in hist.items(): ax.plot(h["val_loss"], label=f"{nm} val"); ax.plot(h["loss"], ls="--", alpha=.5)
    ax.set(xlabel="epoch", ylabel="loss", title="Training (dashed) and validation loss"); ax.legend(); save(fig, path)
