"""Word2Vec CBOW + media dei vettori (documento = media dei vettori delle sue parole).

Parametri di default = Tabella 8 del paper: vector_size 500, window 5, min_count 1,
epochs 10, alpha 0.025, min_alpha 0.0001, CBOW (sg=0), 4 workers.
"""
from __future__ import annotations

import numpy as np


def _to_sentences(token_docs, max_len: int = 500):
    """Spezza i documenti lunghi: gensim tronca le frasi oltre ~10.000 token."""
    for doc in token_docs:
        for i in range(0, len(doc), max_len):
            part = doc[i:i + max_len]
            if part:
                yield part


class Word2VecVectorizer:
    def __init__(self, vector_size=500, window=5, min_count=1, workers=4, epochs=10,
                 alpha=0.025, min_alpha=0.0001, sg=0, seed=42):
        self.params = dict(vector_size=vector_size, window=window, min_count=min_count,
                           workers=workers, epochs=epochs, alpha=alpha,
                           min_alpha=min_alpha, sg=sg, seed=seed)
        self.model = None

    def fit(self, token_docs: list[list[str]]) -> "Word2VecVectorizer":
        from gensim.models import Word2Vec
        sentences = list(_to_sentences(token_docs))
        if not sentences:
            raise ValueError("Corpus vuoto: nessun token su cui addestrare Word2Vec")
        self.model = Word2Vec(sentences=sentences, **self.params)
        return self

    def transform(self, token_docs: list[list[str]]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Chiama prima fit()")
        wv = self.model.wv
        dim = self.params["vector_size"]
        out = np.zeros((len(token_docs), dim), dtype=np.float32)
        for i, doc in enumerate(token_docs):
            idx = [wv.key_to_index[w] for w in doc if w in wv.key_to_index]
            if idx:
                out[i] = wv.vectors[idx].mean(axis=0)
        return out
