"""Regole di masking.

Livelli disponibili:
  none     nessun masking (solo la pulizia comune, vedi clean.py)
  limited  MBTI: solo le menzioni del tipo DELL'UTENTE STESSO
           Big Five: solo i NUMERI dei punteggi (i nomi dei tratti restano)
  full     le regex originali (tipi, funzioni cognitive, vocabolario)
           + mascheramento dei punteggi Big Five

Segnaposto:
  TYPE_MENTION     tipi MBTI e vocabolario (come in PANDORA e nel notebook originale)
  COGFUNC_MENTION  funzioni cognitive
  SCORE_MENTION    numeri dei punteggi Big Five (nuovo)
"""
from __future__ import annotations

import re
from functools import lru_cache

PH_TYPE = "TYPE_MENTION"
PH_FUNC = "COGFUNC_MENTION"
PH_SCORE = "SCORE_MENTION"

# ---------------------------------------------------------------------------
# Regex originali (invariate rispetto al notebook PREPROCESSING__masking)
# ---------------------------------------------------------------------------
TYPE_RE = re.compile(r"\b[EI][NS][TF][JP](?:s|'s)?\b", re.I)
FUNC_RE = re.compile(
    r"(?<![A-Za-z0-9])(?<!Wi-)[NSTF][ie](?:[-/ ]?[NSTF][ie])*(?![A-Za-z0-9])"
)
VOCAB_RE = re.compile(
    r"\b(?:mbti|myers[- ]?briggs|enneagram|socionics|typology|cognitive functions?|"
    r"16 ?personalit(?:y|ies)|personality (?:test|type|quiz)s?|big ?(?:five|5)|"
    r"neo[- ]?(?:pi|ffi)|ipip|truity|hexaco|openness|conscientiousness|"
    r"extraversion|extroversion|agreeableness|neuroticism)\b",
    re.I,
)

# ---------------------------------------------------------------------------
# Punteggi Big Five (nuovo)
# ---------------------------------------------------------------------------
_TRAIT = (
    r"(?:openness(?:\s+to\s+(?:new\s+)?experiences?)?|conscientiousness|extraversion|"
    r"extroversion|agreeableness|neuroticism|emotional\s+stability)"
)
_NUM = r"\d{1,3}(?:\.\d+)?(?!\d)"                  # 85, 4.2, 92.5 (non 1234)
_SUF = r"(?:\s?%|(?:st|nd|rd|th)\b)?"              # 85%, 92nd
_LINK = (
    r"[^\w\n.!?]{0,3}"
    r"(?:(?:scores?|scored|is|was|at|of|around|about|roughly|approx(?:imately)?)"
    r"[^\w\n.!?]{0,3}){0,2}"
)

# "openness: 85", "openness is 85%", "openness to experience - 92nd"
_TRAIT_NUM_RE = re.compile(
    rf"(?P<head>\b{_TRAIT}{_LINK})(?P<num>{_NUM}{_SUF})", re.I
)
# "85% openness", "92nd percentile in openness"
_NUM_TRAIT_RE = re.compile(
    rf"(?<![\w.])(?P<num>{_NUM}{_SUF})"
    rf"(?P<tail>\s*(?:percentiles?\s*)?(?:(?:in|on|for|of)\s+)?(?:the\s+)?{_TRAIT}\b)",
    re.I,
)

# I pattern seguenti sono rischiosi fuori contesto ("N95", "90th percentile" di
# altro): si applicano solo se il commento parla di Big Five.
_CONTEXT_RE = re.compile(
    rf"\b(?:{_TRAIT}|big ?(?:five|5)|rluei|sloan|ipip|neo[- ]?(?:pi|ffi)|hexaco|"
    r"truity|idr ?labs|understand ?myself|qualtrics|personality test)\b",
    re.I,
)
_OCEAN_RE = re.compile(r"\bOCEAN\b")  # solo maiuscolo: "ocean" e' anche l'oceano

_PCTL_BEFORE_RE = re.compile(
    rf"(?<![\w.])(?P<num>{_NUM}{_SUF})(?=\s*percentiles?\b)", re.I
)
_PCTL_AFTER_RE = re.compile(
    rf"(?P<head>\bpercentiles?\s*(?:of|is|was|at|[:=])?\s*)(?P<num>{_NUM}{_SUF})", re.I
)
# "O: 85 C: 40", "O85 C40", "E-22" (maiuscole, case-sensitive)
_LETTER_RE = re.compile(
    rf"(?<![\w'’])(?P<head>[OCEAN](?:\s?[:=\-]\s?)?)(?P<num>{_NUM}{_SUF})"
)
# "85/40/22/75/60" (cinque numeri separati da / , |)
_LIST_RE = re.compile(rf"(?<![\w.]){_NUM}(?:\s*[/,|]\s*{_NUM}){{4}}")
_ANY_NUM_RE = re.compile(r"\d{1,3}(?:\.\d+)?")


def has_bigfive_context(text: str) -> bool:
    return bool(_CONTEXT_RE.search(text) or _OCEAN_RE.search(text))


def mask_bigfive_scores(text: str) -> str:
    """Sostituisce SOLO i numeri dei punteggi Big Five con SCORE_MENTION."""
    ctx = has_bigfive_context(text)
    t = _TRAIT_NUM_RE.sub(lambda m: m.group("head") + PH_SCORE, text)
    t = _NUM_TRAIT_RE.sub(lambda m: PH_SCORE + m.group("tail"), t)
    if ctx:
        t = _PCTL_BEFORE_RE.sub(PH_SCORE, t)
        t = _PCTL_AFTER_RE.sub(lambda m: m.group("head") + PH_SCORE, t)
        t = _LIST_RE.sub(lambda m: _ANY_NUM_RE.sub(PH_SCORE, m.group(0)), t)
        t = _LETTER_RE.sub(lambda m: m.group("head") + PH_SCORE, t)
    return t


# ---------------------------------------------------------------------------
# Livelli
# ---------------------------------------------------------------------------
@lru_cache(maxsize=32)
def _own_type_re(mbti_type: str) -> re.Pattern:
    return re.compile(rf"\b{re.escape(mbti_type)}(?:s|['’]s)?\b", re.I)


def mask_none(text: str) -> str:
    return text


def mask_limited(text: str, own_type: str | None, rules: tuple[str, ...]) -> str:
    """rules: sottoinsieme di ("mbti", "bigfive"), dipende dalle etichette del dataset."""
    if "mbti" in rules and own_type:
        text = _own_type_re(own_type).sub(PH_TYPE, text)
    if "bigfive" in rules:
        text = mask_bigfive_scores(text)
    return text


def mask_full(text: str) -> str:
    """Regex originali + punteggi Big Five.

    L'ordine conta: i punteggi vanno mascherati PRIMA del vocabolario, altrimenti
    "openness 85" diventa "TYPE_MENTION 85" e il numero non e' piu' riconoscibile.
    """
    t = TYPE_RE.sub(PH_TYPE, text)
    t = FUNC_RE.sub(PH_FUNC, t)
    t = mask_bigfive_scores(t)
    t = VOCAB_RE.sub(PH_TYPE, t)
    return t


LEVELS = ("none", "limited", "full")


def mask_text(level: str, text: str, own_type: str | None, rules: tuple[str, ...]) -> str:
    if level == "none":
        return mask_none(text)
    if level == "limited":
        return mask_limited(text, own_type, rules)
    if level == "full":
        return mask_full(text)
    raise ValueError(f"Livello sconosciuto: {level!r} (validi: {LEVELS})")
