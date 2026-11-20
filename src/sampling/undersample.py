"""Random undersampling a livello di UTENTE/testo (vale per qualsiasi modello)."""
from __future__ import annotations

import numpy as np


def undersample_indices(y, ratio: float = 1.0, seed: int = 42) -> np.ndarray:
    """Posizioni da tenere: tutta la minoranza + una parte a caso della maggioranza.

    ratio = minoranza/maggioranza desiderato dopo il taglio (1.0 = classi pari).
    """
    y = np.asarray(y).astype(int)
    counts = np.bincount(y, minlength=2)
    minority = int(np.argmin(counts))
    majority = 1 - minority
    n_keep_major = min(int(counts[majority]), int(np.ceil(counts[minority] / ratio)))
    rng = np.random.default_rng(seed)
    idx_min = np.flatnonzero(y == minority)
    idx_maj = rng.choice(np.flatnonzero(y == majority), size=n_keep_major, replace=False)
    return np.sort(np.concatenate([idx_min, idx_maj]))


def undersample_strata(Y, cap="median", seed: int = 42) -> np.ndarray:
    """Undersampling MULTI-LABEL (per RoBERTa multi-task) per STRATI di etichette.

    Con piu' target insieme non si puo' bilanciare "la classe maggioritaria" di ciascuno.
    Ogni combinazione di etichette (es. il tipo MBTI a 4 lettere = 16 strati, le 5 etichette
    Big Five = 32 strati) e' limitata a `cap` autori: "median" = la dimensione mediana degli
    strati (o un intero). Gli strati piccoli restano interi, quelli grandi sono ridotti a caso.
    Ritorna le posizioni da tenere.
    """
    Y = np.asarray(Y).astype(int)
    keys = np.array(["".join(map(str, r)) for r in Y])
    uniq, counts = np.unique(keys, return_counts=True)
    limit = max(int(np.median(counts)), 1) if cap == "median" else max(int(cap), 1)
    rng = np.random.default_rng(seed)
    keep = []
    for k in uniq:
        idx = np.flatnonzero(keys == k)
        keep.append(idx if len(idx) <= limit else rng.choice(idx, size=limit, replace=False))
    return np.sort(np.concatenate(keep))
