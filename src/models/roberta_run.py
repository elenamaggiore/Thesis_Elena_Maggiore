"""Un run RoBERTa = (livello di masking, strategia, famiglia di target, seed).

Collega i documenti (src/data), le cinque strategie (src/sampling) e il modello (roberta.py).
Usato da scripts/train_roberta.py e scripts/run_roberta_sweep.py.
"""
from __future__ import annotations

from pathlib import Path

from ..data.documents import load_documents
from ..data.targets import TARGETS
from ..sampling.gpt2_aug import CacheOnlyGenerator, SyntheticCache, make_signature
from ..sampling.multitask import build_multitask_dataset
from ..sampling.strategies import STRATEGIES


def run_name(level: str, strategy: str, dataset: str, seed: int, pilot: bool = False) -> str:
    return f"{level}__{strategy}__{dataset}__s{seed}" + ("__pilot" if pilot else "")


def docs_path(docs_dir, level: str, dataset: str) -> Path:
    return Path(docs_dir) / f"documents_{level}_{dataset}.jsonl.gz"


def run_one(level: str, strategy: str, dataset: str, cfg: dict, *, docs_dir, runs_dir,
            cache_path=None, gpt2_model: str = "gpt2", gen_seed: int = 42, max_chunks: int | None = 8,
            limit_authors: int | None = None, docs=None, log=print):
    """Ritorna il riassunto del run, oppure None se era gia' completato."""
    from .roberta import train_run                       # import pesante (torch) solo qui
    pilot = bool(limit_authors)
    out = Path(runs_dir) / run_name(level, strategy, dataset, cfg.get("seed", 42), pilot)
    if (out / "done.json").exists():
        log(f"[{out.name}] gia' completato, salto")
        return None
    if docs is None:
        docs = load_documents(docs_path(docs_dir, level, dataset))
    if limit_authors:                                    # pilota: sottoinsieme casuale e riproducibile
        docs = docs.sample(n=min(limit_authors, len(docs)), random_state=cfg.get("split_seed", 42))
    s = STRATEGIES[strategy]
    cache = reader = None
    if s.gpt2:
        if not cache_path:
            raise ValueError(f"La strategia {strategy} usa GPT-2: serve --cache (cache dei sintetici)")
        cache = SyntheticCache(cache_path)
        reader = CacheOnlyGenerator(make_signature(gpt2_model, seed=gen_seed))
    try:
        data = build_multitask_dataset(docs, TARGETS[dataset], strategy, generator=reader, cache=cache,
                                       split_seed=cfg.get("split_seed", 42), val_size=cfg.get("val_size", 0.2),
                                       max_chunks=max_chunks)
        log(f"[{out.name}] train {data.report['n_train']:,} | val {data.report['n_val']:,} | "
            f"sintetici {data.report['synthetic_in_train']:,} | "
            f"leakage (val con sorgente nel train): {data.report['val_rows_with_source_in_train']}")
        meta = {**cfg, "level": level, "strategy": strategy, "dataset": dataset, "pilot": pilot}
        return train_run(meta, data, out, log=lambda m: log(f"[{out.name}] {m}"))
    finally:
        if cache is not None:
            cache.close()
