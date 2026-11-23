"""
Chunking comments (max 512 tokens) for each author.
Pooling chunks per author to perform casual resampling for each epoch.

"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

TEXT_COL = "text"
LABEL_COL = "y"
ID_COL = "author"

def parse_comments(x) -> tuple[list[str], bool]:
    if not isinstance(x, str):
        return [], False
    s = x.strip()
    if s.startswith("["):
        try:
            v = json.loads(s)
        except ValueError:
            v = None
        if isinstance(v, list):
            return [str(c) for c in v if str(c).strip()], True
    if "|||" in s:
        return [c.strip() for c in s.split("|||") if c.strip()], True
    return ([s] if s else []), False

def read_split(
    path,
    target: str,
    require_label: bool = True,
    allow_generic_label: bool = True
):
    """
    Returns a CSV with raw text, labels, author ids, number of discarded empty rows.

    Train and validation have text, y, author columns. For the test set, only
    columns text, target-specific columns, author are kept.

    """

    df = pd.read_csv(path, low_memory=False)

    if TEXT_COL not in df.columns:
        raise KeyError(
            f"{path}: column '{TEXT_COL}' not found."
            f"Columns: {list(df.columns)}"
        )

    if ID_COL not in df.columns:
        raise KeyError(
            f"{path}: column '{ID_COL}' not found."
            f"Columns: {list(df.columns)}"
        )

    if target in df.columns:
        lcol = target
    elif allow_generic_label and LABEL_COL in df.columns:
        lcol = LABEL_COL
    else:
        lcol = None

    if lcol is None and require_label:
        raise KeyError(
            f"{path}: label column '{target}' not found"
            + (f" né '{LABEL_COL}'" if allow_generic_label else "")
        )

    keep = df[TEXT_COL].map(
        lambda v: len(parse_comments(v)[0]) > 0
    )

    n_drop = int((~keep).sum())
    df = df[keep]

    ids = df[ID_COL].astype(str).tolist()

    labels = None

    if lcol is not None:
        labels = df[lcol].astype(float).round().astype(int).to_numpy()

        if not set(np.unique(labels)) <= {0, 1}:
            raise ValueError(
                f"{path}: label is not binary '{lcol}' "
            )

    return df[TEXT_COL].tolist(), labels, ids, n_drop


def evenly_spaced(n: int, k: int) -> list[int]:
    if n <= k:
        return list(range(n))
    return sorted(set(np.linspace(0, n - 1, k).round().astype(int).tolist()))


def _pack(comment_ids: list[list[int]], body: int, sep_id: int) -> list[list[int]]:
    chunks, cur = [], []
    for ids in comment_ids:
        if not ids:
            continue
        if cur and len(cur) + 1 + len(ids) > body:
            chunks.append(cur)
            cur = []
        if cur:
            cur.append(sep_id)
        cur.extend(ids)
    if cur:
        chunks.append(cur)
    return chunks


def build_chunks(tokenizer, texts, max_len: int, max_chunks: int, max_comments: int = 1500):
    body = max_len - 2
    cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
    out = []
    for raw in texts:
        comments, is_list = parse_comments(raw)
        if len(comments) > max_comments:  # limita la tokenizzazione degli autori molto attivi
            comments = [comments[i] for i in evenly_spaced(len(comments), max_comments)]
        if is_list:
            ids = tokenizer(comments, add_special_tokens=False, truncation=True, max_length=body)["input_ids"]
            chunks = _pack(ids, body, sep_id)
        else:
            toks = tokenizer(comments[0] if comments else "", add_special_tokens=False,
                             truncation=False)["input_ids"] if comments else []
            chunks = [toks[i:i + body] for i in range(0, len(toks), body)]
        chunks = [chunks[i] for i in evenly_spaced(len(chunks), max_chunks)]
        out.append([[cls_id] + c + [sep_id] for c in chunks])
    return out


class ChunkPool:

    def __init__(self, pool: list[list[list[int]]], labels):
        self.pool = pool
        self.labels = np.zeros(len(pool), dtype=int) if labels is None else np.asarray(labels)

    def epoch_size(self, chunks_per_author: int) -> int:
        return sum(min(chunks_per_author, len(c)) for c in self.pool)

    def sample_epoch(self, chunks_per_author: int, rng: np.random.Generator):
        items = []
        for a, chunks in enumerate(self.pool):
            n = len(chunks)
            if n == 0:
                continue
            for i in rng.choice(n, size=min(chunks_per_author, n), replace=False):
                items.append((chunks[i], int(self.labels[a]), a))
        order = rng.permutation(len(items))
        return [items[i] for i in order]

    def eval_items(self):
        return [(c, int(self.labels[a]), a) for a, chunks in enumerate(self.pool) for c in chunks]


def build_pool(tokenizer, texts, labels, max_len: int, max_chunks: int, max_comments: int = 1500) -> ChunkPool:
    return ChunkPool(build_chunks(tokenizer, texts, max_len, max_chunks, max_comments), labels)


class ChunkDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def make_collate(pad_id: int):
    def collate(batch):
        L = max(len(b[0]) for b in batch)
        inp = torch.full((len(batch), L), pad_id, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for i, (ids, _, _) in enumerate(batch):
            inp[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            att[i, :len(ids)] = 1
        return (inp, att, torch.tensor([b[1] for b in batch], dtype=torch.long),
                torch.tensor([b[2] for b in batch], dtype=torch.long))
    return collate
