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
