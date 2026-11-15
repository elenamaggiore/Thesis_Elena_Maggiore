
from __future__ import annotations

import pandas as pd

MBTI_TARGETS = (
    "introverted",
    "intuitive",
    "thinking",
    "perceiving",
)

BIG_FIVE_TARGETS = (
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
)

TARGETS = {
    "mbti": MBTI_TARGETS,
    "big_five": BIG_FIVE_TARGETS,
}

# (lettera del polo "1", posizione nel tipo MBTI)
_MBTI_POLES = {
    "introverted": ("I", 0),
    "intuitive": ("N", 1),
    "thinking": ("T", 2),
    "perceiving": ("P", 3),
}


def add_targets(
    docs: pd.DataFrame,
    labels: dict[str, dict],
    dataset: str,
    bigfive_cutoffs: dict[str, float] | float | None = None,
) -> pd.DataFrame:

    if dataset not in TARGETS:
        raise ValueError(f"Dataset sconosciuto: {dataset!r}")

    docs = docs.copy()

    # Verifica che tutti gli autori abbiano etichette
    missing = set(docs["author"]) - set(labels)

    if missing:
        raise ValueError(
            f"{len(missing)} autori senza etichette valide"
        )

    lab = [labels[a] for a in docs["author"]]

    # --------------------------------------------------
    # MBTI: 4 target binari
    # --------------------------------------------------
    if dataset == "mbti":

        for col, (letter, pos) in _MBTI_POLES.items():
            docs[col] = [
                int(l["mbti"][pos] == letter)
                for l in lab
            ]

    # --------------------------------------------------
    # Big Five: 5 target binari + 5 score originali
    # --------------------------------------------------
    elif dataset == "big_five":

        # Definizione dei cutoff
        if bigfive_cutoffs is None:
            cutoffs = {
                col: 50.0
                for col in BIG_FIVE_TARGETS
            }

        elif isinstance(bigfive_cutoffs, (int, float)):
            cutoffs = {
                col: float(bigfive_cutoffs)
                for col in BIG_FIVE_TARGETS
            }

        elif isinstance(bigfive_cutoffs, dict):
            cutoffs = {
                col: float(bigfive_cutoffs.get(col, 50.0))
                for col in BIG_FIVE_TARGETS
            }

        else:
            raise TypeError(
                "bigfive_cutoffs deve essere un numero, "
                "un dizionario oppure None"
            )

        # Generazione dei target e conservazione degli score
        for col in BIG_FIVE_TARGETS:

            score = [
                float(l[col])
                for l in lab
            ]

            docs[f"{col}_score"] = score

            docs[col] = [
                int(s > cutoffs[col])
                for s in score
            ]

    return docs
