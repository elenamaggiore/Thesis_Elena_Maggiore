"""Etichette binarie (target) a partire dalle etichette degli autori.

MBTI      4 target binari, stessi nomi/polarita' delle colonne di author_profiles.csv:
            introverted = 1 se I (0 se E)     intuitive  = 1 se N (0 se S)
            thinking    = 1 se T (0 se F)     perceiving = 1 se P (0 se J)
Big Five  5 target binari: 1 se il punteggio (percentile 0-100) e' > cutoff, altrimenti 0.
          Il cutoff puo' essere UN NUMERO (uguale per tutti i tratti, default 50: split sulla
          mediana dei percentili, come nel dataset essays) oppure un DIZIONARIO
          {tratto: cutoff} con un cutoff per tratto. Il punteggio grezzo resta nella colonna
          <tratto>_score.
"""
from __future__ import annotations

import pandas as pd

MBTI_TARGETS = ("introverted", "intuitive", "thinking", "perceiving")
BIG_FIVE_TARGETS = ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")

TARGETS = {
    "mbti": MBTI_TARGETS,
    "big_five": BIG_FIVE_TARGETS,
    "final": MBTI_TARGETS + BIG_FIVE_TARGETS,
}

# (lettera del polo "1", posizione nel tipo)
_MBTI_POLES = {"introverted": ("I", 0), "intuitive": ("N", 1), "thinking": ("T", 2), "perceiving": ("P", 3)}


def cutoff_map(cutoffs) -> dict[str, float]:
    """Numero -> lo stesso cutoff per tutti i tratti; dizionario -> un cutoff per tratto."""
    if isinstance(cutoffs, dict):
        missing = [t for t in BIG_FIVE_TARGETS if t not in cutoffs]
        if missing:
            raise ValueError(f"Cutoff mancanti per i tratti: {missing}")
        return {t: float(cutoffs[t]) for t in BIG_FIVE_TARGETS}
    return {t: float(cutoffs) for t in BIG_FIVE_TARGETS}


def add_targets(docs: pd.DataFrame, labels: dict[str, dict], dataset: str,
                bigfive_cutoff=50.0) -> pd.DataFrame:
    """Aggiunge ai documenti le colonne target. `labels` = load_author_types(profiles, dataset)."""
    if dataset not in TARGETS:
        raise ValueError(f"Dataset sconosciuto: {dataset!r}")
    docs = docs.copy()
    missing = [a for a in docs["author"] if a not in labels]
    if missing:
        raise ValueError(f"{len(missing)} autori senza etichette valide (es. {missing[:3]})")
    lab = [labels[a] for a in docs["author"]]
    if dataset in ("mbti", "final"):
        for col, (letter, pos) in _MBTI_POLES.items():
            docs[col] = [int(l["mbti"][pos] == letter) for l in lab]
    if dataset in ("big_five", "final"):
        cut = cutoff_map(bigfive_cutoff)
        for col in BIG_FIVE_TARGETS:
            score = [float(l[col]) for l in lab]
            docs[f"{col}_score"] = score
            docs[col] = [int(x > cut[col]) for x in score]
    return docs
