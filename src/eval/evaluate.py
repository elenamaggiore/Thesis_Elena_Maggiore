"""Valutazione sul test `final`: ensemble di checkpoint, metriche per task e bootstrap appaiato.

Ensemble = model averaging (Christian et al.): si fa la media delle probabilita' per autore
di piu' modelli identici addestrati con seed diversi. Le metriche sono calcolate sulle
probabilita' medie (soglia 0.5).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import (METRICS, bootstrap_indices, bootstrap_metrics, ci95, mean_over_tasks,
                      paired_difference, task_metrics)


def _key(authors) -> str:
    return hashlib.sha1("|".join(map(str, authors)).encode("utf-8")).hexdigest()


def ensemble_predictions(ckpts, test_frame: pd.DataFrame, *, device=None, batch_size: int = 32,
                         recompute: bool = False):
    """Ritorna (autori, P media, config). Le predizioni di ogni checkpoint sono messe in
    <ckpt>/test_probs.npz e riusate se il test e' lo stesso."""
    from ..models.roberta import predict_frame        # import pesante (torch) solo qui
    cfgs = [json.loads((Path(c) / "config.json").read_text()) for c in ckpts]
    ref = cfgs[0]
    for c, cfg in zip(ckpts, cfgs):
        for k in ("level", "strategy", "dataset", "targets"):
            if cfg[k] != ref[k]:
                raise ValueError(f"{c}: {k}={cfg[k]!r} diverso dagli altri checkpoint ({ref[k]!r}): "
                                 f"un ensemble richiede modelli identici su seed diversi")
    per_ckpt = []
    for c in ckpts:
        cache = Path(c) / "test_probs.npz"
        key = _key(test_frame["author"])
        if cache.exists() and not recompute:
            z = np.load(cache, allow_pickle=True)
            if str(z["key"]) == key:
                per_ckpt.append((list(z["authors"]), z["probs"]))
                continue
        authors, P = predict_frame(c, test_frame, device=device, batch_size=batch_size)
        np.savez(cache, authors=np.array(authors), probs=P, key=key)
        per_ckpt.append((authors, P))
    common = [a for a in per_ckpt[0][0] if all(a in set(x[0]) for x in per_ckpt)]
    mats = []
    for authors, P in per_ckpt:
        pos = {a: i for i, a in enumerate(authors)}
        mats.append(P[[pos[a] for a in common]])
    return common, np.mean(mats, axis=0), ref


def evaluate_family(authors, P, test_frame: pd.DataFrame, targets, idx, thr: float = 0.5) -> dict:
    """Metriche per task e media, con IC al 95% dal bootstrap (indici `idx` condivisi)."""
    Y = test_frame.set_index("author").loc[authors, list(targets)].to_numpy().astype(int)
    point = task_metrics(Y, P, targets, thr)
    point_mean = mean_over_tasks(point)
    boot_task, boot_mean = bootstrap_metrics(Y, P, targets, idx, thr)
    return {"targets": list(targets), "n_authors": len(authors), "point": point, "point_mean": point_mean,
            "ci": {t: {m: ci95(boot_task[t][m]) for m in METRICS} for t in targets},
            "ci_mean": {m: ci95(boot_mean[m]) for m in METRICS}, "_boot_mean": boot_mean}


def results_table(res: dict) -> pd.DataFrame:
    rows = []
    for t in res["targets"]:
        for m in METRICS:
            lo, hi = res["ci"][t][m]
            rows.append({"task": t, "metric": m, "value": res["point"][t][m], "ci_low": lo, "ci_high": hi})
    for m in METRICS:
        lo, hi = res["ci_mean"][m]
        rows.append({"task": "MEDIA", "metric": m, "value": res["point_mean"][m], "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def evaluate_checkpoints(ckpts, test_frame, *, pair_with=None, B: int = 2000, boot_seed: int = 0,
                         thr: float = 0.5, device=None, batch_size: int = 32, recompute: bool = False) -> dict:
    """Valuta un ensemble (e, opzionalmente, lo confronta con un secondo ensemble sugli STESSI autori)."""
    authors_a, P_a, cfg_a = ensemble_predictions(ckpts, test_frame, device=device,
                                                 batch_size=batch_size, recompute=recompute)
    out = {"config": {k: cfg_a[k] for k in ("level", "strategy", "dataset", "encoder", "targets")},
           "n_checkpoints": len(ckpts)}
    if pair_with:
        authors_b, P_b, cfg_b = ensemble_predictions(pair_with, test_frame, device=device,
                                                     batch_size=batch_size, recompute=recompute)
        if cfg_b["level"] != cfg_a["level"]:
            raise ValueError("Il confronto appaiato richiede lo stesso livello di masking (stessi autori)")
        common = [a for a in authors_a if a in set(authors_b)]
        pa = {a: i for i, a in enumerate(authors_a)}
        pb = {a: i for i, a in enumerate(authors_b)}
        P_a, P_b = P_a[[pa[a] for a in common]], P_b[[pb[a] for a in common]]
        authors = common
    else:
        authors = authors_a
    idx = bootstrap_indices(len(authors), B, boot_seed)        # lo STESSO ricampionamento per tutti
    out["a"] = evaluate_family(authors, P_a, test_frame, cfg_a["targets"], idx, thr)
    if pair_with:
        out["b_config"] = {k: cfg_b[k] for k in ("level", "strategy", "dataset", "encoder", "targets")}
        out["b"] = evaluate_family(authors, P_b, test_frame, cfg_b["targets"], idx, thr)
        diff = paired_difference(out["a"]["_boot_mean"], out["b"]["_boot_mean"])
        for m in METRICS:                                       # differenza puntuale sulle metriche medie
            diff[m]["diff"] = out["a"]["point_mean"][m] - out["b"]["point_mean"][m]
        out["paired_difference"] = diff
    return out
