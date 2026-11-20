"""RoBERTa multi-task (Shum et al.) con addestramento a livello di CHUNK e validazione per UTENTE.

Modello     encoder pre-addestrato (roberta-base | roberta-large | bert-base-* | bert-large-*)
            -> mean pooling sui token del chunk (maschera di attenzione) -> dropout 0.3
            -> UN layer lineare con tanti output quanti i target: 5 (Big Five) o 4 (MBTI).
            Classificazione binaria multi-label: loss = BCE con pos_weight per task.
Training    chunk fino a 512 token con l'etichetta dell'autore; a ogni epoca si ricampionano
            al piu' `chunks_per_author` chunk per autore dal pool (nessun autore domina).
Validazione a livello UTENTE: media delle probabilita' dei chunk dell'autore (model averaging
            di Christian et al., applicato ai chunk). Epoca migliore = AUC medio sui task.
Ottimizzatore AdamW con lr piu' basso sull'encoder (1e-5, come Shum) e piu' alto sulla testa,
            weight decay 0.01 (non su bias e LayerNorm), gradient clipping 1.0, scheduler con
            warmup del 6% e decadimento lineare, batch 16, 10 epoche, early stopping a 2.
Preprocessing minimo: il testo arriva com'e' (lemmi/stopword rovinano l'input sub-word).
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from ..eval.metrics import auc_score, binary_metrics
from .chunks import ChunkPool, build_pool

DEFAULTS = dict(
    encoder="roberta-base",          # anche: roberta-large (il migliore in Shum), bert-base-uncased, bert-large-uncased
    dropout=0.3,
    lr_encoder=1e-5,                 # Shum: 1e-5 dopo ricerca dell'iperparametro
    lr_head=1e-3,                    # testa piu' alta: parte da zero, l'encoder e' gia' addestrato
    weight_decay=0.01,
    clip=1.0,
    warmup=0.06,
    batch_size=16,
    epochs=10,
    patience=2,
    max_len=512,
    chunks_per_author=4,             # chunk campionati per autore a ogni epoca
    max_pool_chunks=32,              # tetto del pool di chunk per autore (memoria)
    max_eval_chunks=16,              # chunk per autore in validazione/test
    pos_weight_min=0.5,
    pos_weight_max=3.0,
    amp=True,                        # precisione mista, solo su GPU
    seed=42,                         # seed del modello e del ricampionamento
    split_seed=42,                   # seed dello split: fisso, uguale per tutti i membri di un ensemble
    val_size=0.2,
)


class MultiTaskEncoder(nn.Module):
    def __init__(self, encoder: str, n_tasks: int, dropout: float = 0.3, pretrained: bool = True):
        super().__init__()
        if pretrained:
            self.encoder = AutoModel.from_pretrained(encoder)
        else:                                      # in valutazione: i pesi arrivano dal checkpoint
            self.encoder = AutoModel.from_config(AutoConfig.from_pretrained(encoder))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.encoder.config.hidden_size, n_tasks)

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)      # mean pooling
        return self.head(self.dropout(pooled))                          # logit, uno per task


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_count_fn(tok, batch: int = 2048):
    """Conta i token di ogni commento (senza token speciali)."""
    def count(comments: list[str]) -> list[int]:
        out = []
        for i in range(0, len(comments), batch):
            enc = tok(comments[i:i + batch], add_special_tokens=False, truncation=False,
                      verbose=False)["input_ids"]
            out += [len(x) for x in enc]
        return out
    return count


def compute_pos_weight(Y: np.ndarray, lo: float = 0.5, hi: float = 3.0) -> np.ndarray:
    """neg/pos per task, limitato a [lo, hi] per non sbilanciare troppo la loss."""
    pos = Y.sum(axis=0)
    neg = len(Y) - pos
    w = np.where(pos > 0, neg / np.maximum(pos, 1), 1.0)
    return np.clip(w, lo, hi).astype(np.float32)


def _autocast(device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.float16,
                          enabled=bool(enabled and device.type == "cuda"))


def _scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


@torch.no_grad()
def predict_pool(model, tok, pool: ChunkPool, device, max_len: int, batch_size: int, amp: bool = False) -> np.ndarray:
    """Probabilita' per AUTORE (n_autori x n_task): media delle probabilita' dei suoi chunk."""
    model.eval()
    texts, owner = pool.eval_items()
    n_tasks = pool.labels.shape[1]
    probs = np.zeros((len(texts), n_tasks), dtype=np.float32)
    order = np.argsort([len(t) for t in texts])                # chunk simili vicini: meno padding
    for i in range(0, len(order), batch_size):
        idx = order[i:i + batch_size]
        enc = tok([texts[j] for j in idx], truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt")
        with _autocast(device, amp):
            logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        probs[idx] = torch.sigmoid(logits.float()).cpu().numpy()
    sums = np.zeros((len(pool), n_tasks))
    cnt = np.zeros(len(pool))
    np.add.at(sums, owner, probs)
    np.add.at(cnt, owner, 1)
    return sums / np.maximum(cnt, 1)[:, None]


def _param_groups(model, lr_encoder, lr_head, weight_decay):
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight", "LayerNorm.bias")
    groups = []
    for params_of, lr in ((model.encoder, lr_encoder), (model.head, lr_head)):
        named = list(params_of.named_parameters())
        groups.append({"params": [p for n, p in named if not any(k in n for k in no_decay)],
                       "lr": lr, "weight_decay": weight_decay})
        groups.append({"params": [p for n, p in named if any(k in n for k in no_decay)],
                       "lr": lr, "weight_decay": 0.0})
    return [g for g in groups if g["params"]]


def val_metrics(Y: np.ndarray, P: np.ndarray, targets) -> dict:
    auc = [auc_score(Y[:, j], P[:, j]) for j in range(len(targets))]
    f1 = [binary_metrics(Y[:, j], P[:, j])["macro_f1"] for j in range(len(targets))]
    return {"auc": dict(zip(targets, auc)), "mean_auc": float(np.nanmean(auc)) if not np.all(np.isnan(auc)) else float("nan"),
            "mean_macro_f1": float(np.mean(f1))}


def train_run(cfg: dict, data, out_dir, tokenizer=None, log=print) -> dict:
    """Addestra UN modello sui dati `data` (MultiTaskData) e salva il checkpoint migliore in out_dir.

    cfg: parametri (vedi DEFAULTS) piu' i metadati che finiscono in config.json
    (level, strategy, dataset, ...). Ritorna il riassunto salvato in done.json.
    """
    cfg = {**DEFAULTS, **cfg}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = list(data.targets)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg["amp"] and device.type == "cuda")
    tok = tokenizer or AutoTokenizer.from_pretrained(cfg["encoder"])
    count_fn = make_count_fn(tok)
    max_tokens = cfg["max_len"] - 2

    t0 = time.time()
    train_pool = build_pool(data.train, targets, count_fn, max_tokens, cap=cfg["max_pool_chunks"])
    val_pool = build_pool(data.val, targets, count_fn, max_tokens, cap=cfg["max_eval_chunks"])
    log(f"autori train {len(train_pool):,} (val {len(val_pool):,}); chunk per epoca "
        f"{train_pool.epoch_size(cfg['chunks_per_author']):,}  [chunking {time.time() - t0:.0f}s]")

    model = MultiTaskEncoder(cfg["encoder"], len(targets), cfg["dropout"]).to(device)
    pos_w = compute_pos_weight(train_pool.labels, cfg["pos_weight_min"], cfg["pos_weight_max"])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, device=device))
    opt = torch.optim.AdamW(_param_groups(model, cfg["lr_encoder"], cfg["lr_head"], cfg["weight_decay"]))
    steps_per_epoch = math.ceil(train_pool.epoch_size(cfg["chunks_per_author"]) / cfg["batch_size"])
    total = steps_per_epoch * cfg["epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(cfg["warmup"] * total), total)
    scaler = _scaler(use_amp)

    history, best, bad = [], -1.0, 0
    best_epoch = 0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        rng = np.random.default_rng(cfg["seed"] * 1000 + epoch)          # ricampionamento diverso a ogni epoca
        items = train_pool.sample_epoch(rng, cfg["chunks_per_author"])
        losses, t1 = [], time.time()
        for i in range(0, len(items), cfg["batch_size"]):
            batch = items[i:i + cfg["batch_size"]]
            enc = tok([t for t, _ in batch], truncation=True, max_length=cfg["max_len"],
                      padding=True, return_tensors="pt")
            y = torch.tensor(np.stack([l for _, l in batch]), dtype=torch.float32, device=device)
            with _autocast(device, use_amp):
                logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
            loss = loss_fn(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(float(loss.detach()))
        P = predict_pool(model, tok, val_pool, device, cfg["max_len"], cfg["batch_size"], use_amp)
        m = val_metrics(val_pool.labels.astype(int), P, targets)
        score = m["mean_auc"] if not math.isnan(m["mean_auc"]) else -1.0
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **m,
                        "seconds": round(time.time() - t1, 1)})
        log(f"epoca {epoch}/{cfg['epochs']}: loss {np.mean(losses):.4f} | val AUC medio {m['mean_auc']:.4f} "
            f"| macro-F1 {m['mean_macro_f1']:.4f} | {time.time() - t1:.0f}s")
        if score > best + 1e-4:
            best, best_epoch, bad = score, epoch, 0
            torch.save(model.state_dict(), out_dir / "best.pt")
            np.savez(out_dir / "val_probs.npz", authors=np.array(val_pool.authors), probs=P,
                     y=val_pool.labels)
        else:
            bad += 1
            if bad >= cfg["patience"]:
                log(f"early stopping: nessun miglioramento per {cfg['patience']} epoche")
                break

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "config.json").write_text(json.dumps(
        {**cfg, "targets": targets, "n_tasks": len(targets), "pos_weight": pos_w.tolist(),
         "data_report": data.report}, indent=2, default=str))
    summary = {"best_epoch": best_epoch, "best_mean_auc": best, "epochs_run": len(history),
               "seconds": round(time.time() - t0, 1)}
    (out_dir / "done.json").write_text(json.dumps(summary, indent=2))
    return summary


def load_checkpoint(ckpt_dir, device=None):
    """Ricostruisce modello e tokenizer di un run salvato. Ritorna (model, tokenizer, config)."""
    ckpt_dir = Path(ckpt_dir)
    cfg = json.loads((ckpt_dir / "config.json").read_text())
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskEncoder(cfg["encoder"], cfg["n_tasks"], cfg["dropout"], pretrained=False)
    model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location="cpu"))
    return model.to(device).eval(), AutoTokenizer.from_pretrained(cfg["encoder"]), cfg


def predict_frame(ckpt_dir, frame, device=None, batch_size: int = 32):
    """Probabilita' per autore di un checkpoint su un DataFrame di documenti (es. il test `final`).
    Ritorna (autori, probabilita' n_autori x n_task)."""
    model, tok, cfg = load_checkpoint(ckpt_dir, device)
    device = next(model.parameters()).device
    targets = cfg["targets"]
    pool = build_pool(frame, targets, make_count_fn(tok), cfg["max_len"] - 2,
                      cap=cfg["max_eval_chunks"], require_labels=False)
    P = predict_pool(model, tok, pool, device, cfg["max_len"], batch_size, bool(cfg["amp"]))
    return pool.authors, P
