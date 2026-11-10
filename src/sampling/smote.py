"""SMOTE sui VETTORI (parametri = Tabella 9 del paper: strategy 1, k=5, seed 42).

SMOTE interpola tra vettori di lunghezza fissa: funziona sui vettori Word2Vec mediati
(pipeline classica). NON si applica a token, sequenze o testo (RoBERTa, CNN su
sequenze, BERT-LSTM, RAG).
"""
from __future__ import annotations

import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler


def oversample_vectors(X, y, method: str = "smote", sampling_strategy: float = 1.0,
                       k_neighbors: int = 5, seed: int = 42):
    """Ritorna (X_res, y_res, info). method: "smote" | "random" | "none"."""
    y = np.asarray(y).astype(int)
    counts = np.bincount(y, minlength=2)
    info = {"method_requested": method, "before": counts.tolist()}
    n_min = int(counts.min())
    if method == "none" or n_min == int(counts.max()):
        info.update(method_used="none", after=counts.tolist())
        return X, y, info
    if method == "smote" and n_min >= 2:
        k = min(k_neighbors, n_min - 1)               # SMOTE richiede k < n_minoranza
        sampler = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k, random_state=seed)
        info["method_used"], info["k_neighbors"] = "smote", k
    elif method in ("smote", "random"):
        sampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=seed)
        info["method_used"] = "random" + (" (fallback: troppo pochi esempi per SMOTE)" if method == "smote" else "")
    else:
        raise ValueError(f"Metodo sconosciuto: {method!r}")
    X_res, y_res = sampler.fit_resample(X, y)
    info["after"] = np.bincount(y_res, minlength=2).tolist()
    return X_res, y_res, info
