"""I 20 dataset (4 livelli di masking x 5 strategie) in versione MULTI-TASK, per RoBERTa.

RoBERTa e' un unico modello con piu' uscite (5 Big Five o 4 MBTI): serve UN solo split e
UN solo training set con tutte le etichette. Le cinque strategie diventano:

  paper_leaky      GPT-2 su TUTTI gli autori -> split di righe (reali + sintetici mescolati)   [leakage]
  noleak_full      split per autore -> GPT-2 solo sul train (input preprocessato "full")
  noleak_minimal   split per autore -> GPT-2 solo sul train (input preprocessato "minimal")
  no_oversampling  split per autore, train com'e'
  undersampling    split per autore -> undersampling per STRATI di etichette sul train

  - SMOTE non esiste qui: agisce solo sui vettori Word2Vec, non sul testo che legge RoBERTa.
    Per RoBERTa le strategie 1-3 differiscono quindi solo per leakage e per l'input di GPT-2.
  - Un documento sintetico (generato per la minoranza di UN target) eredita TUTTE le
    etichette del suo autore: i sintetici dei vari target si uniscono (un autore una volta).
  - Lo squilibrio residuo e' gestito dalla pos_weight della loss (vedi models/roberta.py).
  - Il testo dei documenti reali NON e' preprocessato (lemmi/stopword rovinano l'input di
    RoBERTa): il preprocessing della strategia riguarda solo l'input di GPT-2.
I sintetici vanno prima generati e messi in cache (scripts/generate_synthetic.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .gpt2_aug import augment_minority, make_input_fn
from .strategies import STRATEGIES
from .undersample import undersample_strata

MASKING_LEVELS = ("none", "limited", "full", "llm")


def all_datasets(levels=MASKING_LEVELS, strategies=None) -> list[tuple[str, str]]:
    """I 20 dataset: (livello di masking, strategia)."""
    return [(lv, st) for lv in levels for st in (strategies or list(STRATEGIES))]


def multilabel_split(Y, val_size: float = 0.2, seed: int = 42):
    """Split stratificato sulla COMBINAZIONE delle etichette (gli strati con meno di 2
    autori sono uniti in uno strato 'rare'). Se non e' possibile, ripiega sul primo target
    e, in ultimo, su uno split casuale."""
    Y = np.asarray(Y).astype(int)
    n = len(Y)
    keys = np.array(["".join(map(str, r)) for r in Y])
    uniq, counts = np.unique(keys, return_counts=True)
    keys = np.where(np.isin(keys, uniq[counts < 2]), "rare", keys)
    if (keys == "rare").sum() == 1:                       # uno strato 'rare' da solo non e' valido
        keys[keys == "rare"] = uniq[np.argmax(counts)]
    for strat in (keys, Y[:, 0], None):
        try:
            return train_test_split(np.arange(n), test_size=val_size, stratify=strat, random_state=seed)
        except ValueError:
            continue
    raise ValueError("Split impossibile")


@dataclass
class MultiTaskData:
    train: pd.DataFrame
    val: pd.DataFrame
    targets: list[str]
    report: dict


def _counts(frame: pd.DataFrame, targets) -> dict:
    return {t: {"pos": int((frame[t] == 1).sum()), "neg": int((frame[t] == 0).sum())} for t in targets}


def build_multitask_dataset(docs: pd.DataFrame, targets, strategy, *, generator=None, cache=None,
                            split_seed: int = 42, val_size: float = 0.2, gpt2_multiplier: float = 1.0,
                            chunk_size: int = 5, max_chunks: int | None = 8,
                            undersample_cap="median", gpt2_input: str | None = None) -> MultiTaskData:
    """docs: documenti (uno per autore) di UN livello di masking; targets: colonne da predire."""
    s = STRATEGIES[strategy] if isinstance(strategy, str) else strategy
    targets = list(targets)
    docs = docs.reset_index(drop=True).copy()
    if "source_author" not in docs:
        docs["source_author"] = docs["author"]
    if "is_synthetic" not in docs:
        docs["is_synthetic"] = False
    input_fn, input_tag = make_input_fn(gpt2_input or s.preprocessing) if s.gpt2 else (None, "raw")

    def synth(frame: pd.DataFrame) -> pd.DataFrame:
        """Sintetici per la minoranza di OGNI target, uniti (un autore una volta sola)."""
        frame = frame.reset_index(drop=True)
        parts = [augment_minority(frame, t, generator, multiplier=gpt2_multiplier, chunk_size=chunk_size,
                                  max_chunks=max_chunks, cache=cache, seed=split_seed,
                                  input_fn=input_fn, input_tag=input_tag) for t in targets]
        return pd.concat(parts, ignore_index=True).drop_duplicates("author")

    rep = {"strategy": s.name, "targets": targets, "n_real": int(len(docs)),
           "gpt2_input": input_tag if s.gpt2 else None}
    if s.leakage:
        # come nel paper: augmentation su TUTTI, poi split di righe (LEAKAGE)
        allrows = pd.concat([docs, synth(docs)], ignore_index=True)
        tr, va = multilabel_split(allrows[targets].to_numpy(), val_size, split_seed)
        train, val = allrows.iloc[tr], allrows.iloc[va]
    else:
        tr, va = multilabel_split(docs[targets].to_numpy(), val_size, split_seed)
        train, val = docs.iloc[tr], docs.iloc[va]
        rep["train_before_balancing"] = int(len(train))
        if s.undersample:
            train = train.iloc[undersample_strata(train[targets].to_numpy(), undersample_cap, split_seed)]
        if s.gpt2:
            train = pd.concat([train, synth(train)], ignore_index=True)
    train, val = train.reset_index(drop=True), val.reset_index(drop=True)
    rep.update(n_train=int(len(train)), n_val=int(len(val)),
               synthetic_in_train=int(train["is_synthetic"].sum()),
               val_synthetic_rows=int(val["is_synthetic"].sum()),
               # leakage: righe di validation la cui sorgente compare anche nel train (0 senza leakage)
               val_rows_with_source_in_train=int(val["source_author"].isin(set(train["source_author"])).sum()),
               train_labels=_counts(train, targets), val_labels=_counts(val, targets))
    return MultiTaskData(train=train, val=val, targets=targets, report=rep)
