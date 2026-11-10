"""Preprocessing del testo.

Due livelli, entrambi restituiscono una LISTA DI TOKEN:

  full     preprocessing del paper (Patel & Dhavale, sez. 3.4): minuscolo, contrazioni
           espanse, link/menzioni/URL tolti, parole-tipo MBTI tolte, solo caratteri
           alfabetici, lemmatizzazione, stopword tolte.
  minimal  solo minuscolo e tokenizzazione: niente lemmi, niente stopword, numeri e
           punteggiatura interna tenuti. E' il livello adatto ai modelli con tokenizer
           proprio (RoBERTa, BERT), che il preprocessing "full" danneggerebbe.

"""
from __future__ import annotations

import re
from functools import lru_cache

from ..masking.rules import TYPE_RE

_PLACEHOLDERS = {
    "TYPE_MENTION": "typemention",
    "COGFUNC_MENTION": "cogfuncmention",
    "SCORE_MENTION": "scoremention",
}
KEEP_STOPWORDS = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves",
                  "you", "your", "yours", "yourself", "yourselves"}

_APOS_RE = re.compile(r"[’‘`´]")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
_MENTION_RE = re.compile(r"[@#]\w+")
_NONALPHA_RE = re.compile(r"[^a-z]+")
_MIN_TOKEN_RE = re.compile(r"[a-z0-9_]+(?:'[a-z]+)?")

_SPECIAL = {"won't": "will not", "can't": "cannot", "shan't": "shall not",
            "ain't": "is not", "let's": "let us", "y'all": "you all"}
_S_WORDS = ("it", "that", "what", "he", "she", "there", "here", "who", "where",
            "how", "when", "why")
_SUFFIXES = (("n't", " not"), ("'re", " are"), ("'ve", " have"),
             ("'ll", " will"), ("'d", " would"), ("'m", " am"))


def expand_contractions(text: str) -> str:
    """don't -> do not, can't -> cannot, I'm -> I am, it's -> it is (i possessivi restano)."""
    t = _APOS_RE.sub("'", text)
    for k, v in _SPECIAL.items():
        t = re.sub(rf"\b{re.escape(k)}\b", v, t, flags=re.I)
    t = re.sub(rf"\b({'|'.join(_S_WORDS)})'s\b", r"\1 is", t, flags=re.I)
    for suf, rep in _SUFFIXES:
        t = re.sub(rf"(?<=\w){re.escape(suf)}\b", rep, t, flags=re.I)
    return t


def _ensure_nltk():
    import nltk
    for path, name in (("corpora/wordnet", "wordnet"), ("corpora/stopwords", "stopwords"),
                       ("corpora/omw-1.4", "omw-1.4")):
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


class FullPreprocessor:
    def __init__(self):
        _ensure_nltk()
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        self._stop = set(stopwords.words("english")) - KEEP_STOPWORDS
        self._lemma = lru_cache(maxsize=200_000)(WordNetLemmatizer().lemmatize)

    def __call__(self, text: str) -> list[str]:
        t = text
        for ph, tok in _PLACEHOLDERS.items():      # protegge i segnaposto prima del minuscolo
            t = t.replace(ph, tok)
        t = expand_contractions(t.lower())
        t = _URL_RE.sub(" ", t)
        t = _MENTION_RE.sub(" ", t)
        t = TYPE_RE.sub(" ", t)                    # parole-tipo MBTI (anche plurali)
        t = _NONALPHA_RE.sub(" ", t)               # solo caratteri alfabetici
        out = []
        for w in t.split():
            w = self._lemma(w)
            if w in self._stop or (len(w) == 1 and w != "i"):
                continue
            out.append(w)
        return out


def preprocess_minimal(text: str) -> list[str]:
    return _MIN_TOKEN_RE.findall(_APOS_RE.sub("'", text).lower())


@lru_cache(maxsize=None)
def get_preprocessor(kind: str):
    """kind: "full" | "minimal"  ->  funzione  testo -> lista di token."""
    if kind == "full":
        return FullPreprocessor()
    if kind == "minimal":
        return preprocess_minimal
    raise ValueError(f"Preprocessing sconosciuto: {kind!r} (validi: 'full', 'minimal')")
