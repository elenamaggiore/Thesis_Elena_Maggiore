"""Pulizia comune a TUTTI i livelli di masking (compreso "none").

Tenerla separata dal masking garantisce che i livelli differiscano
solo per il masking e non per altre operazioni.
"""
from __future__ import annotations

import html
import re

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
BOT_RE = re.compile(
    r"i am a bot|i'm a bot|this action was performed automatically|"
    r"^\*\*?beep boop|^this (?:comment|post) was (?:removed|deleted)",
    re.I,
)
DELETED = {"[deleted]", "[removed]", "deleted", "removed"}


def clean_comment(raw) -> str | None:
    """Restituisce il testo pulito, oppure None se il commento va scartato."""
    if not isinstance(raw, str):
        return None
    t = html.unescape(raw).strip()
    if not t or t.lower() in DELETED or BOT_RE.search(t):
        return None
    # citazioni di altri utenti ("> ...")
    t = " ".join(line for line in t.splitlines() if not line.lstrip().startswith(">"))
    t = URL_RE.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not any(c.isalpha() for c in t):
        return None
    return t


def prepare_comment(raw, min_words: int, max_words: int):
    """Pulizia + filtro lunghezza + troncamento.

    Ritorna (testo, motivo): motivo e' None se il commento e' tenuto, altrimenti
    "bad" (vuoto/bot/cancellato) o "short" (meno di min_words parole).
    Il troncamento vale per TUTTI i commenti (nel notebook originale i commenti
    "leaky" non venivano troncati).
    """
    t = clean_comment(raw)
    if t is None:
        return None, "bad"
    words = t.split()
    if len(words) < min_words:
        return None, "short"
    if len(words) > max_words:
        t = " ".join(words[:max_words])
    return t, None
