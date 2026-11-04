"""Applicazione del masking LLM ai file di commenti.

Differenze rispetto a pipeline.py (regex):
  - l'LLM e' lento: solo i commenti "candidati" (prefiltro ad alto recall) vengono
    mandati al modello; gli altri restano identici al livello "none";
  - ogni risposta del modello e' salvata in una cache SQLite: se il processo si
    interrompe, rilanciando il comando si riparte da dove era arrivato;
  - le righe tenute sono ESATTAMENTE le stesse degli altri livelli (stessa pulizia),
    quindi i file sono confrontabili riga per riga.
Output: <out_dir>/<livello>/comments_<dataset>.csv   (livello = "llm" oppure "llm_hybrid")
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sqlite3
from collections import Counter
from pathlib import Path

import pandas as pd

from .clean import prepare_comment
from .llm_core import (PROMPT_VERSION, apply_spans, build_messages, is_candidate,
                       parse_model_output, validate_spans)
from .pipeline import DATASETS
from .rules import mask_full


class SpanCache:
    """Cache SQLite: (modello, versione prompt, testo) -> span restituiti dal modello."""

    def __init__(self, path: Path, model_tag: str):
        self.model_tag = model_tag
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.execute("CREATE TABLE IF NOT EXISTS spans (key TEXT PRIMARY KEY, spans TEXT)")
        self.db.commit()

    def key(self, text: str) -> str:
        raw = f"{PROMPT_VERSION}|{self.model_tag}|{text}".encode("utf-8")
        return hashlib.sha1(raw).hexdigest()

    def get_many(self, keys: list[str]) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {}
        for i in range(0, len(keys), 500):
            part = keys[i:i + 500]
            q = ",".join("?" * len(part))
            for k, v in self.db.execute(f"SELECT key, spans FROM spans WHERE key IN ({q})", part):
                out[k] = json.loads(v)
        return out

    def put_many(self, items: dict[str, list[dict]]):
        self.db.executemany(
            "INSERT OR REPLACE INTO spans (key, spans) VALUES (?, ?)",
            [(k, json.dumps(v, ensure_ascii=False)) for k, v in items.items()],
        )
        self.db.commit()

    def close(self):
        self.db.close()


def _safe_tag(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", name)


def process_dataset_llm(
    name: str,
    data_dir: Path,
    out_dir: Path,
    backend,                       # None solo con dry_run=True
    model_name: str,
    hybrid: bool = False,
    chunk_size: int = 50_000,
    min_words: int = 5,
    max_words: int = 500,
    batch_size: int = 32,
    force: bool = False,
    dry_run: bool = False,
    limit: int | None = None,      # pilota: ferma dopo N commenti mandati al modello
    audit_n: int = 200,
) -> dict | None:
    spec = DATASETS[name]
    src = Path(data_dir) / spec["file"]
    if not src.exists():
        print(f"[{name}] file non trovato: {src}")
        return None

    level = "llm_hybrid" if hybrid else "llm"
    if limit:
        level += "_pilot"
    target = Path(out_dir) / level / spec["file"]
    if not dry_run and target.exists() and not force:
        print(f"[{name}] output gia' presente: {target} (usa --force per rifarlo)")
        return None

    cache = None
    if not dry_run:
        cache = SpanCache(Path(out_dir) / f"llm_cache_{_safe_tag(model_name)}.sqlite",
                          _safe_tag(model_name))
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".csv.tmp")
        tmp.unlink(missing_ok=True)

    c = Counter()
    seen_unique: set[str] = set()
    audit: list[dict] = []
    rnd = random.Random(0)
    first = True
    stop = False

    try:
        for ci, chunk in enumerate(
            pd.read_csv(src, chunksize=chunk_size, dtype=str, keep_default_na=False)
        ):
            c["rows_in"] += len(chunk)
            keep, cleaned = [], []
            for j, raw in enumerate(chunk["body"].tolist()):
                text, why = prepare_comment(raw, min_words, max_words)
                if why:
                    c[why] += 1
                    continue
                keep.append(j)
                cleaned.append(text)
            c["rows_kept"] += len(keep)
            if not keep:
                continue

            cand_idx = [k for k, t in enumerate(cleaned) if is_candidate(t)]
            c["candidates"] += len(cand_idx)

            if dry_run:
                seen_unique.update(cleaned[k] for k in cand_idx)
                print(f"[{name}] chunk {ci + 1}: {c['rows_kept']:,} tenute, "
                      f"{c['candidates']:,} candidate ({len(seen_unique):,} uniche)", flush=True)
                continue

            # --- cache + chiamate al modello (solo testi unici non in cache) ---
            keys = {k: cache.key(cleaned[k]) for k in cand_idx}
            cached = cache.get_many(list(set(keys.values())))
            todo = {}                                  # chiave -> testo
            for k in cand_idx:
                if keys[k] not in cached:
                    todo[keys[k]] = cleaned[k]
            c["cache_hits"] += len(set(keys.values())) - len(todo)

            todo_items = list(todo.items())
            if limit is not None:
                room = max(limit - c["llm_calls"], 0)
                todo_items = todo_items[:room]
            for i in range(0, len(todo_items), batch_size):
                part = todo_items[i:i + batch_size]
                outs = backend.generate([build_messages(t) for _, t in part])
                new = {}
                for (key, text), raw in zip(part, outs):
                    c["llm_calls"] += 1
                    spans = parse_model_output(raw)
                    if spans is None:                  # non in cache: verra' ritentato
                        c["parse_errors"] += 1
                        continue
                    new[key] = spans
                if new:
                    cache.put_many(new)
                    cached.update(new)
                print(f"[{name}] chunk {ci + 1}: {c['llm_calls']:,} chiamate al modello "
                      f"(errori di parsing: {c['parse_errors']})", flush=True)

            # --- applicazione degli span (codice nostro, non dell'LLM) ---
            masked = list(cleaned)
            for k in cand_idx:
                spans = cached.get(keys[k])
                if spans is None:
                    c["candidates_without_answer"] += 1
                    continue
                accepted, rej = validate_spans(cleaned[k], spans)
                c["spans_accepted"] += len(accepted)
                c["spans_rejected"] += rej
                m = apply_spans(cleaned[k], accepted)
                if hybrid:
                    m = mask_full(m)
                masked[k] = m
                if m != cleaned[k]:
                    c["changed"] += 1
                    if len(audit) < audit_n:
                        audit.append({"original": cleaned[k], "masked": m,
                                      "spans": json.dumps(accepted, ensure_ascii=False)})
                    elif rnd.random() < audit_n / max(c["changed"], 1):
                        audit[rnd.randrange(audit_n)] = {
                            "original": cleaned[k], "masked": m,
                            "spans": json.dumps(accepted, ensure_ascii=False)}

            out = chunk.iloc[keep].copy()
            out["body"] = masked
            out.to_csv(tmp, mode="w" if first else "a", header=first, index=False,
                       encoding="utf-8")
            first = False

            if limit is not None and c["llm_calls"] >= limit:
                stop = True
                break
        if not dry_run:
            if first:                                   # nessuna riga scritta
                tmp.write_text("", encoding="utf-8")
            os.replace(tmp, target)
    except BaseException:
        if not dry_run:
            tmp.unlink(missing_ok=True)
        raise
    finally:
        if cache is not None:
            cache.close()

    stats = {
        "dataset": name, "level": level, "model": model_name, "prompt_version": PROMPT_VERSION,
        "hybrid": hybrid, "limit": limit, "stopped_by_limit": stop, "dry_run": dry_run,
        "rows_in": c["rows_in"], "rows_kept": c["rows_kept"],
        "dropped_bad": c["bad"], "dropped_short": c["short"],
        "candidates": c["candidates"],
        "pct_candidates": round(100 * c["candidates"] / max(c["rows_kept"], 1), 3),
    }
    if dry_run:
        stats["unique_candidates"] = len(seen_unique)
        print(f"[{name}] DRY RUN: {stats['candidates']:,} candidate "
              f"({stats['pct_candidates']}% delle righe tenute), "
              f"{stats['unique_candidates']:,} testi unici = chiamate al modello necessarie")
        return stats

    stats.update({
        "llm_calls": c["llm_calls"], "cache_hits": c["cache_hits"],
        "parse_errors": c["parse_errors"],
        "candidates_without_answer": c["candidates_without_answer"],
        "spans_accepted": c["spans_accepted"], "spans_rejected": c["spans_rejected"],
        "changed": c["changed"],
        "pct_changed": round(100 * c["changed"] / max(c["rows_kept"], 1), 3),
    })
    Path(out_dir, f"llm_stats_{name}{'_pilot' if limit else ''}.json").write_text(
        json.dumps(stats, indent=2))
    if audit:
        pd.DataFrame(audit).to_csv(
            Path(out_dir) / f"llm_audit_{name}{'_pilot' if limit else ''}.csv",
            index=False, encoding="utf-8-sig")
    print(f"[{name}] completato: {c['changed']:,} commenti modificati "
          f"({stats['pct_changed']}%). Output: {target}")
    return stats
