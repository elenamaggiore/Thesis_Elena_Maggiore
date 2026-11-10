"""Un DOCUMENTO per autore, a partire dai file di commenti (gia' mascherati).

Gli autori PANDORA hanno migliaia di commenti: ne teniamo al massimo `max_comments`
per autore, scelti a caso (reservoir sampling con seed fisso, quindi riproducibile
e senza caricare in memoria tutti i commenti). Lo stesso tetto vale per tutti gli
esperimenti, cosi' il confronto resta equo.
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from .targets import add_targets

COMMENT_SEP = " ||| "   # separatore tra commenti nel campo "text"


def build_user_documents(comments_csv: Path, authors: set[str], max_comments: int = 200,
                         seed: int = 42, chunk_size: int = 200_000) -> pd.DataFrame:
    rnd = random.Random(seed)
    reservoir: dict[str, list[str]] = {}
    seen: dict[str, int] = {}
    for chunk in pd.read_csv(comments_csv, usecols=["author", "body"], dtype=str,
                             keep_default_na=False, chunksize=chunk_size):
        chunk = chunk[chunk["author"].isin(authors)]
        for a, b in zip(chunk["author"].tolist(), chunk["body"].tolist()):
            if not b:
                continue
            n = seen.get(a, 0) + 1
            seen[a] = n
            lst = reservoir.setdefault(a, [])
            if len(lst) < max_comments:
                lst.append(b)
            else:
                j = rnd.randrange(n)
                if j < max_comments:
                    lst[j] = b
    rows = [{"author": a, "comments": reservoir[a], "n_comments_seen": seen[a],
             "text": COMMENT_SEP.join(reservoir[a])} for a in sorted(reservoir)]
    return pd.DataFrame(rows)


def build_documents(comments_csv: Path, labels: dict[str, dict], dataset: str,
                    max_comments: int = 200, seed: int = 42, bigfive_cutoff: dict[str, float]={"openness": 50,"conscientiousness": 50,"extraversion": 50,"agreeableness": 50,"neuroticism": 50},
                    chunk_size: int = 200_000) -> pd.DataFrame:
    """Documenti + target binari. Tiene solo gli autori con etichette valide e almeno un commento."""
    docs = build_user_documents(comments_csv, set(labels), max_comments, seed, chunk_size)
    docs = add_targets(docs, labels, dataset, bigfive_cutoff)
    docs["source_author"] = docs["author"]
    docs["is_synthetic"] = False
    return docs


def save_documents(docs: pd.DataFrame, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    docs.to_json(path, orient="records", lines=True, compression="gzip", force_ascii=False)


def load_documents(path: Path) -> pd.DataFrame:
    return pd.read_json(path, orient="records", lines=True, compression="gzip", dtype=False)
