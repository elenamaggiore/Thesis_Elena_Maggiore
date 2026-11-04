"""Applicazione del masking ai file di commenti, a blocchi (chunk).

Ogni dataset viene letto UNA volta; la pulizia e' calcolata una volta e poi
applicata a tutti i livelli richiesti. I file originali non vengono mai toccati:
l'output va in  <out_dir>/<livello>/<stesso nome del file di input>.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import pandas as pd

from .clean import prepare_comment
from .rules import LEVELS, mask_text

# Quali regole di masking "limited" valgono per ciascun dataset (dipende dalle
# etichette che gli utenti hanno).
DATASETS = {
    "mbti": {"file": "comments_mbti.csv", "rules": ("mbti",)},
    "big_five": {"file": "comments_big_five.csv", "rules": ("bigfive",)},
    "final": {"file": "comments_final.csv", "rules": ("mbti", "bigfive")},
}


def load_author_types(profiles_path: Path) -> dict[str, str]:
    """author -> tipo MBTI (maiuscolo, a 4 lettere); scarta valori mancanti/invalidi."""
    df = pd.read_csv(
        profiles_path, usecols=["author", "mbti"], dtype=str, keep_default_na=False
    )
    df["mbti"] = df["mbti"].str.strip().str.upper()
    valid = df["mbti"].str.fullmatch(r"[EI][NS][TF][JP]")
    return dict(zip(df.loc[valid, "author"], df.loc[valid, "mbti"]))


def process_dataset(
    name: str,
    data_dir: Path,
    out_dir: Path,
    levels: list[str],
    author_types: dict[str, str],
    chunk_size: int = 200_000,
    min_words: int = 5,
    max_words: int = 500,
    force: bool = False,
) -> dict | None:
    spec = DATASETS[name]
    rules = spec["rules"]
    src = Path(data_dir) / spec["file"]
    if not src.exists():
        print(f"[{name}] file non trovato: {src}")
        return None

    targets = {lv: Path(out_dir) / lv / spec["file"] for lv in levels}
    todo = [lv for lv in levels if force or not targets[lv].exists()]
    if not todo:
        print(f"[{name}] niente da fare (output gia' presenti, usa --force per rifare)")
        return None

    tmp = {lv: targets[lv].with_suffix(".csv.tmp") for lv in todo}
    for lv in todo:
        targets[lv].parent.mkdir(parents=True, exist_ok=True)
        tmp[lv].unlink(missing_ok=True)

    shared = Counter()                       # righe lette / scartate (uguali per ogni livello)
    changed = {lv: 0 for lv in todo}         # commenti modificati dal masking
    no_type = 0                              # righe MBTI senza tipo noto per l'autore
    first = True

    try:
        for i, chunk in enumerate(
            pd.read_csv(src, chunksize=chunk_size, dtype=str, keep_default_na=False)
        ):
            shared["rows_in"] += len(chunk)
            bodies = chunk["body"].tolist()
            authors = chunk["author"].tolist()

            keep, cleaned = [], []
            for j, raw in enumerate(bodies):
                text, why = prepare_comment(raw, min_words, max_words)
                if why:
                    shared[why] += 1
                    continue
                keep.append(j)
                cleaned.append(text)
            shared["rows_kept"] += len(keep)
            if not keep:
                continue

            base = chunk.iloc[keep]
            own = [author_types.get(authors[j]) for j in keep]
            if "mbti" in rules:
                no_type += sum(1 for o in own if o is None)

            for lv in todo:
                masked = [mask_text(lv, t, o, rules) for t, o in zip(cleaned, own)]
                changed[lv] += sum(1 for m, t in zip(masked, cleaned) if m != t)
                out = base.copy()
                out["body"] = masked
                out.to_csv(
                    tmp[lv], mode="w" if first else "a", header=first,
                    index=False, encoding="utf-8",
                )
            first = False
            print(f"[{name}] chunk {i + 1}: {shared['rows_in']:,} righe lette, "
                  f"{shared['rows_kept']:,} tenute", flush=True)

        for lv in todo:                       # sposta i file solo a lavoro finito
            os.replace(tmp[lv], targets[lv])
    except BaseException:
        for lv in todo:
            tmp[lv].unlink(missing_ok=True)
        raise

    stats = {
        "dataset": name,
        "rules_limited": list(rules),
        "params": {"min_words": min_words, "max_words": max_words},
        "rows_in": shared["rows_in"],
        "rows_kept": shared["rows_kept"],
        "dropped_bad": shared["bad"],
        "dropped_short": shared["short"],
        "rows_without_known_type": no_type if "mbti" in rules else None,
        "changed_by_level": changed,
        "pct_changed_by_level": {
            lv: round(100 * changed[lv] / max(shared["rows_kept"], 1), 3) for lv in todo
        },
    }
    stats_path = Path(out_dir) / f"masking_stats_{name}.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"[{name}] completato. Statistiche: {stats_path}")
    return stats
