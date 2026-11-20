"""Chunk di testo (fino a 512 token) con l'etichetta dell'AUTORE, e pool per autore.

Un documento (i commenti di un autore) e' diviso in chunk impacchettando commenti consecutivi
fino a max_tokens (510 = 512 meno <s> e </s>). Un commento piu' lungo diventa un chunk da solo
(il tokenizer lo tronca). In training, a ogni epoca si ricampionano al piu' k chunk per autore
dal pool, SENZA reinserimento: nessun autore domina, qualunque sia il suo numero di commenti.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def build_chunks(comments: list[str], counts: list[int], max_tokens: int = 510, sep: str = " ") -> list[str]:
    chunks, cur, cur_n = [], [], 0
    for c, n in zip(comments, counts):
        if not c or not c.strip():
            continue
        n = max(int(n), 1)
        if cur and cur_n + n > max_tokens:
            chunks.append(sep.join(cur))
            cur, cur_n = [], 0
        cur.append(c)
        cur_n += n
    if cur:
        chunks.append(sep.join(cur))
    return chunks


def evenly_spaced(n: int, cap: int | None) -> list[int]:
    """Al piu' `cap` indici distribuiti su tutto l'intervallo (deterministico)."""
    if cap is None or n <= cap:
        return list(range(n))
    return sorted(set(np.linspace(0, n - 1, cap).round().astype(int).tolist()))


@dataclass
class ChunkPool:
    authors: list[str]
    chunks: list[list[str]]          # chunks[i] = chunk dell'autore i
    labels: np.ndarray               # (n_autori, n_task) float32

    def __len__(self):
        return len(self.authors)

    def epoch_size(self, k: int) -> int:
        return sum(min(k, len(c)) for c in self.chunks)

    def sample_epoch(self, rng: np.random.Generator, k: int) -> list[tuple[str, np.ndarray]]:
        """Al piu' k chunk per autore, senza reinserimento; ordine mescolato."""
        items = []
        for i, pool in enumerate(self.chunks):
            take = len(pool) if len(pool) <= k else k
            for j in rng.choice(len(pool), size=take, replace=False):
                items.append((pool[int(j)], self.labels[i]))
        order = rng.permutation(len(items))
        return [items[j] for j in order]

    def eval_items(self) -> tuple[list[str], np.ndarray]:
        """Tutti i chunk del pool, con l'indice dell'autore di ciascuno."""
        texts, owner = [], []
        for i, pool in enumerate(self.chunks):
            texts += pool
            owner += [i] * len(pool)
        return texts, np.asarray(owner, dtype=int)


def build_pool(frame: pd.DataFrame, targets, count_fn, max_tokens: int = 510,
               cap: int | None = None, require_labels: bool = True) -> ChunkPool:
    """frame: documenti con colonne author, comments (lista) e target. cap: tetto di chunk per autore."""
    authors, chunks, labels = [], [], []
    for _, row in frame.iterrows():
        comments = [str(c) for c in row["comments"]]
        ch = build_chunks(comments, count_fn(comments), max_tokens)
        ch = [ch[i] for i in evenly_spaced(len(ch), cap)]
        if not ch:
            continue
        authors.append(row["author"])
        chunks.append(ch)
        labels.append([row[t] for t in targets] if require_labels else [0] * len(targets))
    return ChunkPool(authors, chunks, np.asarray(labels, dtype=np.float32).reshape(len(authors), len(list(targets))))
