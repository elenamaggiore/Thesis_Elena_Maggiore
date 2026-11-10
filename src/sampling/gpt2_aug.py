"""Augmentation testuale con GPT-2 (Patel & Dhavale, sez. 3.5-3.6).


"""
from __future__ import annotations

import hashlib
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from ..features.preprocess import expand_contractions, get_preprocessor

SEP = " ||| "
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
_FORUM_RE = re.compile(r"\[/?(?:quote|img|url|b|i|u|spoiler|video)[^\]]*\]|&gt;|&amp;|\|\|\|", re.I)


def make_input_fn(kind: str):
    """kind: "full" | "minimal" | "raw" -> (funzione commento->testo per GPT-2, etichetta per la cache)."""
    if kind == "raw":
        return None, "raw"
    pre = get_preprocessor(kind)
    return (lambda c: " ".join(pre(c))), kind


def make_chunks(comments: list[str], chunk_size: int = 5, max_chunks: int | None = None) -> list[str]:
    comments = [c for c in comments if c and c.strip()]
    chunks = [" ".join(comments[i:i + chunk_size]) for i in range(0, len(comments), chunk_size)]
    return chunks[:max_chunks] if max_chunks else chunks


def postprocess_generated(text: str) -> str:
    t = expand_contractions(text or "")
    t = _URL_RE.sub(" ", t)
    t = _FORUM_RE.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


# ---------------------------------------------------------------------------
# Generatori
# ---------------------------------------------------------------------------
def make_signature(model_name="gpt2", context_window=150, min_total_length=200, num_beams=5,
                   temperature=0.7, top_k=50, top_p=0.95, no_repeat_ngram_size=2, seed=42) -> str:
    """Firma dei parametri di generazione: fa parte della chiave di cache."""
    return (f"{model_name}|ctx{context_window}|len{min_total_length}|b{num_beams}|"
            f"T{temperature}|k{top_k}|p{top_p}|nr{no_repeat_ngram_size}|seed{seed}")


class CacheOnlyGenerator:
    """Non genera nulla: serve a LEGGERE la cache creata da generate_synthetic.py
    (con la stessa firma di parametri). Se manca qualcosa, dà errore."""

    def __init__(self, signature: str):
        self.signature = signature

    def generate(self, texts):
        raise RuntimeError(f"{len(texts)} blocchi da generare ma non in cache (firma {self.signature!r}): "
                           f"lancia prima scripts/generate_synthetic.py con gli stessi parametri")


class GPT2Generator:
    """GPT-2 locale (transformers + torch, meglio su GPU). generate(list[str]) -> list[str]."""

    def __init__(self, model_name="gpt2", batch_size=8, context_window=150, min_total_length=200,
                 num_beams=5, temperature=0.7, top_k=50, top_p=0.95, no_repeat_ngram_size=2,
                 seed=42, device=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.manual_seed(seed)
        self._torch = torch
        self.batch_size, self.context_window, self.min_total_length = batch_size, context_window, min_total_length
        self.gen_kwargs = dict(do_sample=True, num_beams=num_beams, temperature=temperature,
                               top_k=top_k, top_p=top_p, no_repeat_ngram_size=no_repeat_ngram_size)
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device).eval()
        self.signature = make_signature(model_name, context_window, min_total_length, num_beams,
                                        temperature, top_k, top_p, no_repeat_ngram_size, seed)

    def generate(self, texts: list[str]) -> list[str]:
        results = [""] * len(texts)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))   # meno padding
        n_batches = (len(order) + self.batch_size - 1) // self.batch_size
        for b, start in enumerate(range(0, len(order), self.batch_size)):
            idx = order[start:start + self.batch_size]
            enc = self.tok([texts[i] for i in idx], return_tensors="pt", padding=True,
                           truncation=True, max_length=self.context_window).to(self.device)
            in_len = enc["input_ids"].shape[1]
            max_length = min(max(self.min_total_length, in_len + 50), 1024)
            with self._torch.no_grad():
                out = self.model.generate(**enc, max_length=max_length,
                                          pad_token_id=self.tok.eos_token_id, **self.gen_kwargs)
            for i, d in zip(idx, self.tok.batch_decode(out[:, in_len:], skip_special_tokens=True)):
                results[i] = d
            if (b + 1) % 20 == 0 or b + 1 == n_batches:
                print(f"  GPT-2: {b + 1}/{n_batches} batch", flush=True)
        return results


class MockGenerator:
    """Per i test: fn(testo) -> testo generato."""
    signature = "mock"

    def __init__(self, fn):
        self.fn, self.calls = fn, 0

    def generate(self, texts):
        self.calls += len(texts)
        return [self.fn(t) for t in texts]


# ---------------------------------------------------------------------------
# Cache (SQLite): un documento sintetico per (autore, numero di copia, parametri)
# ---------------------------------------------------------------------------
class SyntheticCache:
    def __init__(self, path: Path | str | None = None):
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(":memory:" if path is None else str(path))
        self.db.execute("CREATE TABLE IF NOT EXISTS syn (key TEXT PRIMARY KEY, text TEXT)")
        self.db.commit()

    @staticmethod
    def key(signature: str, chunk_size: int, max_chunks, author: str, copy_idx: int, text: str) -> str:
        raw = f"v1|{signature}|{chunk_size}|{max_chunks}|{author}|{copy_idx}|" \
              f"{hashlib.sha1(text.encode('utf-8')).hexdigest()}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def get(self, key: str):
        row = self.db.execute("SELECT text FROM syn WHERE key=?", (key,)).fetchone()
        return None if row is None else row[0]

    def put_many(self, items: dict[str, str]):
        self.db.executemany("INSERT OR REPLACE INTO syn (key, text) VALUES (?, ?)", list(items.items()))
        self.db.commit()

    def close(self):
        self.db.close()


# ---------------------------------------------------------------------------
# Augmentation della classe minoritaria
# ---------------------------------------------------------------------------
def plan_synthetic(y, multiplier: float = 1.0, seed: int = 42):
    """Ritorna [(posizione, copia)] dei documenti da cui generare un sintetico."""
    y = np.asarray(y).astype(int)
    counts = np.bincount(y, minlength=2)
    n_min, n_maj = int(counts.min()), int(counts.max())
    if n_min == n_maj or n_min == 0:
        return []
    positions = np.flatnonzero(y == int(np.argmin(counts)))
    remaining = min(int(round(multiplier * n_min)), n_maj - n_min)
    rng = np.random.default_rng(seed)
    picks, copy_idx = [], 0
    while remaining > 0:
        take = min(remaining, n_min)
        chosen = positions if take == n_min else rng.choice(positions, size=take, replace=False)
        picks += [(int(p), copy_idx) for p in sorted(chosen)]
        remaining -= take
        copy_idx += 1
    return picks


def augment_minority(docs: pd.DataFrame, target: str, generator, *, multiplier: float = 1.0,
                     chunk_size: int = 5, max_chunks: int | None = 8,
                     cache: SyntheticCache | None = None, seed: int = 42,
                     input_fn=None, input_tag: str = "raw") -> pd.DataFrame:
    """Documenti SINTETICI per la classe minoritaria di `target` (da aggiungere a `docs`).

    Le righe sintetiche copiano TUTTE le etichette dell'autore di partenza e hanno
    author = "<autore>#syn<copia>", source_author = autore di partenza, is_synthetic = True.
    input_fn / input_tag: come trasformare ogni commento prima di darlo a GPT-2 (vedi
    make_input_fn); l'etichetta fa parte della chiave di cache.
    Se `generator` e' None, usa solo la cache e dà errore se manca qualcosa.
    """
    docs = docs.reset_index(drop=True)
    picks = plan_synthetic(docs[target].to_numpy(), multiplier, seed)
    if not picks:
        return docs.iloc[0:0].copy()
    cache = cache or SyntheticCache()
    sig = f"{getattr(generator, 'signature', 'none')}|in:{input_tag}"

    keys, found, todo = {}, {}, []
    for pos, c in picks:
        row = docs.iloc[pos]
        k = cache.key(sig, chunk_size, max_chunks, row["author"], c, row["text"])
        keys[(pos, c)] = k
        hit = cache.get(k)
        if hit is not None:
            found[(pos, c)] = hit
        else:
            todo.append((pos, c))

    if todo:
        if generator is None:
            raise RuntimeError(f"{len(todo)} sintetici mancanti in cache e nessun generatore: "
                               f"lancia prima scripts/generate_synthetic.py")
        flat, owner = [], []
        for pos, c in todo:
            src = list(docs.iloc[pos]["comments"])
            if input_fn is not None:
                src = [input_fn(c) for c in src]
            for ch in make_chunks(src, chunk_size, max_chunks):
                flat.append(ch)
                owner.append((pos, c))
        outs = generator.generate(flat) if flat else []
        per_doc: dict[tuple, list[str]] = {}
        for own, out in zip(owner, outs):
            t = postprocess_generated(out)
            if t:
                per_doc.setdefault(own, []).append(t)
        new = {}
        for own in todo:
            text = SEP.join(per_doc.get(own, []))
            found[own] = text
            new[keys[own]] = text
        cache.put_many(new)

    rows = []
    for pos, c in picks:
        text = found[(pos, c)]
        if not text:
            continue                                   # generazione vuota: salta
        r = docs.iloc[pos].to_dict()
        r["author"] = f"{r['author']}#syn{c}"
        r["comments"] = [x.strip() for x in text.split(SEP.strip())]
        r["text"] = text
        r["is_synthetic"] = True
        rows.append(r)
    return pd.DataFrame(rows, columns=docs.columns) if rows else docs.iloc[0:0].copy()
