"""Metriche per task binari e bootstrap APPAIATO sugli autori.

Metriche per task: accuracy, balanced accuracy, macro-F1, F1 della classe positiva,
MCC, AUC (soglia 0.5 sulle probabilita' medie per autore).

Bootstrap appaiato: si ricampionano gli AUTORI del test con reinserimento e lo STESSO
ricampionamento vale per tutti i task e per entrambe le famiglie di modelli (Big Five e
MBTI, che sul `final` hanno gli stessi autori). Cosi' gli intervalli di confidenza al 95%
sono confrontabili e si puo' calcolare l'intervallo della DIFFERENZA tra le due famiglie.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "f1_pos", "mcc", "auc")


def auc_score(y, p) -> float:
    y = np.asarray(y).astype(int)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(p)                                   # gestisce i pareggi (rango medio)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def binary_metrics(y, p, thr: float = 0.5) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    n = tp + tn + fp + fn
    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    tnr = tn / (tn + fp) if (tn + fp) else float("nan")
    f1_pos = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    f1_neg = 2 * tn / (2 * tn + fn + fp) if (2 * tn + fn + fp) else 0.0
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "accuracy": (tp + tn) / n if n else float("nan"),
        "balanced_accuracy": float(np.mean([tpr, tnr])) if not (np.isnan(tpr) or np.isnan(tnr)) else float("nan"),
        "macro_f1": (f1_pos + f1_neg) / 2,
        "f1_pos": f1_pos,
        "mcc": float((tp * tn - fp * fn) / denom) if denom > 0 else 0.0,
        "auc": auc_score(y, p),
    }


def task_metrics(Y, P, tasks, thr: float = 0.5) -> dict[str, dict[str, float]]:
    """Y, P: matrici (autori x task)."""
    return {t: binary_metrics(Y[:, j], P[:, j], thr) for j, t in enumerate(tasks)}


def mean_over_tasks(per_task: dict[str, dict[str, float]]) -> dict[str, float]:
    return {m: float(np.nanmean([v[m] for v in per_task.values()])) for m in METRICS}


def bootstrap_indices(n: int, B: int = 2000, seed: int = 0) -> np.ndarray:
    """Matrice (B x n) di indici di autori: da riusare per TUTTI i task e le due famiglie."""
    return np.random.default_rng(seed).integers(0, n, size=(B, n))


def bootstrap_metrics(Y, P, tasks, idx, thr: float = 0.5):
    """Ritorna (per_task, media): per_task[task][metrica] = array(B); media[metrica] = array(B)."""
    B = idx.shape[0]
    per_task = {t: {m: np.full(B, np.nan) for m in METRICS} for t in tasks}
    for b in range(B):
        yb, pb = Y[idx[b]], P[idx[b]]
        for j, t in enumerate(tasks):
            r = binary_metrics(yb[:, j], pb[:, j], thr)
            for m in METRICS:
                per_task[t][m][b] = r[m]
    mean = {m: np.nanmean(np.vstack([per_task[t][m] for t in tasks]), axis=0) for m in METRICS}
    return per_task, mean


def ci95(samples) -> tuple[float, float]:
    s = np.asarray(samples, dtype=float)
    s = s[~np.isnan(s)]
    if len(s) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def paired_difference(mean_a: dict, mean_b: dict) -> dict[str, dict[str, float]]:
    """Differenza A - B delle metriche medie, sullo stesso ricampionamento degli autori."""
    out = {}
    for m in METRICS:
        d = mean_a[m] - mean_b[m]
        lo, hi = ci95(d)
        out[m] = {"diff": float(np.nanmean(d)), "ci_low": lo, "ci_high": hi,
                  "significant": bool(lo > 0 or hi < 0)}
    return out
