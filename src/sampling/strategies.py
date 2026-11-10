"""Le cinque strategie di sampling dell'esperimento.

Ogni strategia e' un insieme di scelte indipendenti:
  leakage        l'augmentation avviene PRIMA dello split (come nel paper): i testi
                 sintetici di un autore possono finire in validation insieme/contro l'originale
  preprocessing  "full" (preprocessing del paper) oppure "minimal" (solo minuscolo + token)
  gpt2           augmentation testuale della classe minoritaria con GPT-2
  smote          SMOTE sui vettori (solo pipeline Word2Vec)
  undersample    random undersampling della classe maggioritaria (sul train)

Scalini (ogni passo cambia UNA cosa rispetto al precedente):
  paper_leaky     -> noleak_full      : si toglie il leakage
  noleak_full     -> noleak_minimal   : si toglie il preprocessing del paper
  no_oversampling e undersampling     : riferimenti senza GPT-2 ne' SMOTE
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Strategy:
    name: str
    leakage: bool
    preprocessing: str
    gpt2: bool
    smote: bool
    undersample: bool
    description: str

    def __post_init__(self):
        if self.leakage and not self.gpt2:
            raise ValueError("leakage ha senso solo con gpt2 (e' l'augmentation prima dello split)")
        if self.undersample and (self.gpt2 or self.smote):
            raise ValueError("undersample non si combina con gpt2/smote")
        if self.preprocessing not in ("full", "minimal"):
            raise ValueError(f"preprocessing sconosciuto: {self.preprocessing!r}")


STRATEGIES = {
    "paper_leaky": Strategy(
        "paper_leaky", leakage=True, preprocessing="full", gpt2=True, smote=True, undersample=False,
        description="Fedele al paper: GPT-2 prima dello split, preprocessing completo, SMOTE"),
    "noleak_full": Strategy(
        "noleak_full", leakage=False, preprocessing="full", gpt2=True, smote=True, undersample=False,
        description="Split per autore, GPT-2 solo sul train, preprocessing completo, SMOTE"),
    "noleak_minimal": Strategy(
        "noleak_minimal", leakage=False, preprocessing="minimal", gpt2=True, smote=True, undersample=False,
        description="Come noleak_full ma con preprocessing minimo"),
    "no_oversampling": Strategy(
        "no_oversampling", leakage=False, preprocessing="full", gpt2=False, smote=False, undersample=False,
        description="Nessun bilanciamento"),
    "undersampling": Strategy(
        "undersampling", leakage=False, preprocessing="full", gpt2=False, smote=False, undersample=True,
        description="Random undersampling della classe maggioritaria (sul train)"),
}
