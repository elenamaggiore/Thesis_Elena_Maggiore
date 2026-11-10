"""Applica UNA strategia di sampling a UN target di UN dataset.

Input   documenti (uno per autore, con le colonne target) — vedi src/data/documents.py
Output  PreparedTarget:
          train / val         DataFrame a LIVELLO TESTO (author, text, y, is_synthetic, ...)
                              -> per RoBERTa, CNN, BERT-LSTM, RAG
          X_train, y_train_vec, X_val, y_val_vec
                              vettori Word2Vec (con SMOTE se la strategia lo prevede)
                              -> per la pipeline classica (baseline Word2Vec + RF, ecc.)
          transform_texts()   stessa pipeline di feature per nuovi testi (es. il test `final`)
          report              conteggi per classe a ogni passo + diagnostica del leakage

Ordine delle operazioni
  SENZA leakage :  split per autore -> (undersampling) -> GPT-2 sul solo train -> feature -> SMOTE
  CON leakage   :  GPT-2 su TUTTI -> split di righe (reali + sintetici mescolati) -> feature -> SMOTE
Word2Vec e' addestrato sul solo train in tutti i casi: il paper descrive prima lo split e poi
le feature, quindi l'unico leakage riprodotto e' quello dell'augmentation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data.split import stratified_split
from ..features.preprocess import get_preprocessor
from ..features.word2vec import Word2VecVectorizer
from .gpt2_aug import SyntheticCache, augment_minority, make_input_fn
from .smote import oversample_vectors
from .strategies import STRATEGIES, Strategy
from .undersample import undersample_indices


def _counts(y) -> dict:
    y = np.asarray(y).astype(int)
    return {"n": int(len(y)), "pos": int((y == 1).sum()), "neg": int((y == 0).sum())}


@dataclass
class PreparedTarget:
    strategy: str
    target: str
    train: pd.DataFrame
    val: pd.DataFrame
    report: dict
    X_train: np.ndarray | None = None
    y_train_vec: np.ndarray | None = None
    X_val: np.ndarray | None = None
    y_val_vec: np.ndarray | None = None
    _preprocess: object = field(default=None, repr=False)
    _vectorizer: object = field(default=None, repr=False)

    def transform_texts(self, texts) -> np.ndarray:
        """Vettori per nuovi testi (es. il test set `final`), con la STESSA pipeline del train."""
        if self._vectorizer is None:
            raise RuntimeError("Feature non calcolate (prepare_target(..., features=False))")
        return self._vectorizer.transform([self._preprocess(t) for t in texts])


def _as_text_frame(df: pd.DataFrame, target: str) -> pd.DataFrame:
    out = df.copy()
    out["y"] = out[target].astype(int)
    return out.reset_index(drop=True)


def prepare_target(docs: pd.DataFrame, target: str, strategy, *, generator=None,
                   cache: SyntheticCache | None = None, seed: int = 42, val_size: float = 0.2,
                   gpt2_multiplier: float = 1.0, chunk_size: int = 5, max_chunks: int | None = 8,
                   features: bool = True, preprocessing: str | None = None,
                   gpt2_input: str | None = None,
                   w2v_params: dict | None = None, smote_params: dict | None = None) -> PreparedTarget:
    s: Strategy = STRATEGIES[strategy] if isinstance(strategy, str) else strategy
    prep_kind = preprocessing or s.preprocessing
    docs = docs.reset_index(drop=True).copy()
    if "source_author" not in docs:
        docs["source_author"] = docs["author"]
    if "is_synthetic" not in docs:
        docs["is_synthetic"] = False
    # Input di GPT-2: per default lo STESSO preprocessing della strategia ("full"/"minimal");
    # gpt2_input="raw" lo da' non pulito.
    input_fn, input_tag = make_input_fn(gpt2_input or prep_kind) if s.gpt2 else (None, "raw")
    aug_kw = dict(multiplier=gpt2_multiplier, chunk_size=chunk_size, max_chunks=max_chunks,
                  cache=cache, seed=seed, input_fn=input_fn, input_tag=input_tag)
    report = {"strategy": s.name, "target": target, "preprocessing": prep_kind,
              "gpt2_input": input_tag if s.gpt2 else None,
              "real_docs": _counts(docs[target])}

    if s.leakage:
        # --- come nel paper: augmentation su TUTTI, poi split di righe (LEAKAGE) ---
        syn = augment_minority(docs, target, generator, **aug_kw)
        allrows = pd.concat([docs, syn], ignore_index=True)
        report["synthetic_added"] = int(len(syn))
        tr, va = stratified_split(allrows[target], val_size, seed)
        train, val = allrows.iloc[tr], allrows.iloc[va]
    else:
        # --- split per autore PRIMA di qualunque augmentation ---
        tr, va = stratified_split(docs[target], val_size, seed)
        train, val = docs.iloc[tr], docs.iloc[va]
        report["train_before_balancing"] = _counts(train[target])
        if s.undersample:
            keep = undersample_indices(train[target], seed=seed)
            train = train.iloc[keep]
            report["train_after_undersampling"] = _counts(train[target])
        if s.gpt2:
            syn = augment_minority(train.reset_index(drop=True), target, generator, **aug_kw)
            train = pd.concat([train, syn], ignore_index=True)
            report["synthetic_added"] = int(len(syn))

    train, val = _as_text_frame(train, target), _as_text_frame(val, target)
    report["train_text_level"] = _counts(train["y"])
    report["val"] = _counts(val["y"])
    report["val_synthetic_rows"] = int(val["is_synthetic"].sum())
    # DIAGNOSTICA DEL LEAKAGE: righe di validation il cui autore di partenza compare anche nel train.
    # Con la strategia senza leakage deve essere SEMPRE 0.
    report["val_rows_with_source_in_train"] = int(val["source_author"].isin(set(train["source_author"])).sum())

    result = PreparedTarget(strategy=s.name, target=target, train=train, val=val, report=report)

    if features:
        pre = get_preprocessor(prep_kind)
        tok_train = [pre(t) for t in train["text"]]
        vec = Word2VecVectorizer(**(w2v_params or {})).fit(tok_train)
        X_train, y_train = vec.transform(tok_train), train["y"].to_numpy()
        if s.smote:
            X_train, y_train, info = oversample_vectors(X_train, y_train, **(smote_params or {}), seed=seed)
            report["smote"] = info
        result.X_train, result.y_train_vec = X_train, y_train
        result.X_val = vec.transform([pre(t) for t in val["text"]])
        result.y_val_vec = val["y"].to_numpy()
        result._preprocess, result._vectorizer = pre, vec
        report["train_vectors"] = _counts(y_train)
    elif s.smote:
        report["smote"] = "non applicato (features=False): SMOTE agisce solo sui vettori"
    return result
