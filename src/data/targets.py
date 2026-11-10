"""Etichette binarie (target) a partire dalle etichette degli autori.

MBTI      4 target binari, stessi nomi/polarita' delle colonne di author_profiles.csv:
            introverted = 1 se I (0 se E)     intuitive  = 1 se N (0 se S)
            thinking    = 1 se T (0 se F)     perceiving = 1 se P (0 se J)
Big Five  5 target binari: 1 se il punteggio (percentile 0-100) e' > cutoff, altrimenti 0.
          Il cutoff di default e' 50 (split sulla mediana dei percentili). Per tenere anche il punteggio grezzo, la colonna
          <tratto>_score contiene il valore originale.
"""
from __future__ import annotations

import pandas as pd

MBTI_TARGETS = ("introverted", "intuitive", "thinking", "perceiving")
BIG_FIVE_TARGETS = ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")

TARGETS = {
    "mbti": MBTI_TARGETS,
    "big_five": BIG_FIVE_TARGETS,
}

# (lettera del polo "1", posizione nel tipo)
_MBTI_POLES = {"introverted": ("I", 0), "intuitive": ("N", 1), "thinking": ("T", 2), "perceiving": ("P", 3)}


def add_targets(docs: pd.DataFrame, labels: dict[str, dict],
                dataset: str,
                bigfive_cutoffs: dict[str, float] | None = None) -> pd.DataFrame:

    if dataset not in TARGETS:
        raise ValueError(f"Dataset sconosciuto: {dataset!r}")

    docs = docs.copy()

    missing = [a for a in docs["author"] if a not in labels]
    if missing:
        raise ValueError(f"{len(missing)} autori senza etichette valide")

    lab = [labels[a] for a in docs["author"]]

    if dataset in ("mbti", "final"):
        for col, (letter, pos) in _MBTI_POLES.items():
            docs[col] = [int(l["mbti"][pos] == letter) for l in lab]

    if dataset in ("big_five", "final"):
        cutoffs = bigfive_cutoffs or {
            col: 50.0 for col in BIG_FIVE_TARGETS
        }

        for col in BIG_FIVE_TARGETS:
            score = [float(l[col]) for l in lab]
            docs[f"{col}_score"] = score
            docs[col] = [
                int(s > cutoffs[col]) for s in score
            ]

    return docs
