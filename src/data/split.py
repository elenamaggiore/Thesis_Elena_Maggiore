"""Split stratificato train/validation (sulle POSIZIONI delle righe)."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


def stratified_split(y, val_size: float = 0.2, seed: int = 42):
    """Ritorna (idx_train, idx_val), stratificato sul target binario y.

    Per lo split SENZA leakage va chiamato su documenti REALI (un autore = una riga),
    prima di qualunque augmentation: cosi' lo split e' per autore. Per riprodurre il
    paper (augmentation PRIMA dello split) va chiamato su reali + sintetici mescolati.
    """
    y = np.asarray(y).astype(int)
    counts = np.bincount(y)
    if len(counts) < 2 or counts.min() < 2:
        raise ValueError(f"Servono almeno 2 esempi per classe per lo split stratificato "
                         f"(conteggi: {counts.tolist()})")
    return train_test_split(np.arange(len(y)), test_size=val_size, stratify=y, random_state=seed)
