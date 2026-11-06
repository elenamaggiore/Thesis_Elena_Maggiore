"""Applicazione del masking ai file di commenti, a blocchi (chunk).

Ogni dataset viene letto UNA volta; la pulizia e' calcolata una volta e poi
applicata a tutti i livelli richiesti. I file originali non vengono mai toccati:
l'output va in  <out_dir>/<livello>/<stesso nome del file di input>.

ATTENZIONE: i livelli in DROP_SUBREDDIT_LEVELS (oggi solo "full") scartano per intero
i commenti scritti nei subreddit della lista di esclusione. Quei livelli hanno quindi
MENO righe degli altri: il confronto riga-per-riga tra livelli non e' piu' valido,
va fatto per chiave (autore + testo) o dopo aver filtrato anche gli altri file.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import pandas as pd

from .clean import prepare_comment
from .rules import DROP_SUBREDDIT_LEVELS, mask_text, normalize_subreddit

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
    excluded_subreddits: set[str] | None = None,
) -> dict | None:
    spec = DATASETS[name]
    rules = spec["rules"]
    excluded = set(excluded_subreddits or ())
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

    shared = Counter()                       # righe lette / scartate dalla pulizia (uguali per ogni livello)
    changed = {lv: 0 for lv in todo}         # commenti modificati dal masking
    rows_out = {lv: 0 for lv in todo}        # righe scritte per livello
    dropped_sub = {lv: 0 for lv in todo}     # righe scartate per subreddit in lista
    first = {lv: True for lv in todo}
    no_type = 0                              # righe MBTI senza tipo noto per l'autore
    cols = None

    try:
        for i, chunk in enumerate(
            pd.read_csv(src, chunksize=chunk_size, dtype=str, keep_default_na=False)
        ):
            if cols is None:
                cols = list(chunk.columns)
                if excluded and "subreddit" not in cols:
                    raise ValueError(
                        f"{src.name}: manca la colonna 'subreddit', impossibile filtrare "
                        f"i subreddit esclusi")
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

            if excluded:
                in_excluded = [normalize_subreddit(s) in excluded
                               for s in base["subreddit"].tolist()]
            else:
                in_excluded = [False] * len(keep)

            for lv in todo:
                if lv in DROP_SUBREDDIT_LEVELS:
                    sel = [k for k, flag in enumerate(in_excluded) if not flag]
                    dropped_sub[lv] += len(keep) - len(sel)
                else:
                    sel = list(range(len(keep)))
                if not sel:
                    continue
                masked = [mask_text(lv, cleaned[k], own[k], rules) for k in sel]
                changed[lv] += sum(1 for m, k in zip(masked, sel) if m != cleaned[k])
                out = base.iloc[sel].copy()
                out["body"] = masked
                out.to_csv(
                    tmp[lv], mode="w" if first[lv] else "a", header=first[lv],
                    index=False, encoding="utf-8",
                )
                first[lv] = False
                rows_out[lv] += len(sel)
            print(f"[{name}] chunk {i + 1}: {shared['rows_in']:,} righe lette, "
                  f"{shared['rows_kept']:,} tenute", flush=True)

        for lv in todo:
            if first[lv] and cols is not None:   # nessuna riga scritta: crea il file con la sola intestazione
                pd.DataFrame(columns=cols).to_csv(tmp[lv], index=False, encoding="utf-8")
        for lv in todo:                          # sposta i file solo a lavoro finito
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
        "excluded_subreddits_n": len(excluded),
        "rows_out_by_level": rows_out,
        "dropped_excluded_subreddit_by_level": dropped_sub,
        "changed_by_level": changed,
        "pct_changed_by_level": {
            lv: round(100 * changed[lv] / max(rows_out[lv], 1), 3) for lv in todo
        },
    }
    stats_path = Path(out_dir) / f"masking_stats_{name}.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"[{name}] completato. Statistiche: {stats_path}")
    return stats
