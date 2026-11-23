"""
Single-target RoBERTa model for binary classification.
Inspired from Shum et al.' paper.
Training is chunk-based (chunks are labelled with author's label). Validation is
author-based (averages chunk probabilities).

"""
from __future__ import annotations

import gc
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from chunks import ChunkDataset, build_pool, make_collate, read_split

METRICS = ["accuracy", "balanced_accuracy", "macro_f1", "f1_pos", "mcc", "auc"]


@dataclass
class Config:
    encoder: str = "roberta-large"
    dropout: float = 0.3
    lr_encoder: float = 1e-5
    lr_head: float = 1e-3
    weight_decay: float = 0.01
    clip: float = 1.0
    warmup: float = 0.06
    batch_size: int = 16
    epochs: int = 10
    patience: int = 2
    max_len: int = 512
    chunks_per_author: int = 4
    max_pool_chunks: int = 32
    max_eval_chunks: int = 16
    pos_weight_min: float = 0.5
    pos_weight_max: float = 3.0
    amp: bool = True
    seed: int = 11
    split_seed: int = 11
    val_size: float = 0.2

    micro_batch: int = 16              # < batch_size => gradient accumulation
    grad_checkpointing: bool = True   # Turn FALSE if more memory is needed
    max_comments_per_author: int = 1500
    max_authors: int = 0               # for quick testing (0 gets all authors)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class BinaryEncoder(nn.Module):

    def __init__(self, encoder: str, dropout: float, n_out: int = 1, grad_checkpointing: bool = False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder)
        if grad_checkpointing:
            self.encoder.gradient_checkpointing_enable()
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.encoder.config.hidden_size, n_out)
        self.n_out = n_out

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)  # media dei vettori dei token (pad esclusi)
        out = self.head(self.dropout(pooled))
        return out.squeeze(-1) if self.n_out == 1 else out


def val_metrics(y, p, thr: float = 0.5) -> dict:
    """Metrics at author level. Best epoch is epoch with highest AUC (averaged by task)."""
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    yh = (p >= thr).astype(int)
    tp, tn = float(((yh == 1) & (y == 1)).sum()), float(((yh == 0) & (y == 0)).sum())
    fp, fn = float(((yh == 1) & (y == 0)).sum()), float(((yh == 0) & (y == 1)).sum())
    n = len(y)
    tpr = tp / (tp + fn) if tp + fn else np.nan
    tnr = tn / (tn + fp) if tn + fp else np.nan
    f1p = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    f1n = 2 * tn / (2 * tn + fn + fp) if 2 * tn + fn + fp else 0.0
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    npos = float(y.sum())
    nneg = n - npos
    auc = (rankdata(p)[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg) if npos and nneg else np.nan
    return {"n": n, "pos_rate": float(y.mean()), "majority_acc": float(max(y.mean(), 1 - y.mean())),
            "accuracy": (tp + tn) / n,
            "balanced_accuracy": float(np.nanmean([tpr, tnr])) if not (np.isnan(tpr) and np.isnan(tnr)) else np.nan,
            "macro_f1": (f1p + f1n) / 2, "f1_pos": f1p,
            "mcc": float((tp * tn - fp * fn) / den) if den else 0.0, "auc": float(auc)}


@torch.no_grad()
def predict_pool(model, items, n_authors: int, device, amp: bool, batch_size: int, pad_id: int) -> np.ndarray:
    model.eval()
    loader = DataLoader(ChunkDataset(items), batch_size=batch_size, shuffle=False, collate_fn=make_collate(pad_id))
    s, c = np.zeros(n_authors), np.zeros(n_authors)
    for inp, att, _, a in loader:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            logits = model(inp.to(device), att.to(device))
        np.add.at(s, a.numpy(), torch.sigmoid(logits.float()).cpu().numpy())
        np.add.at(c, a.numpy(), 1)
    return np.where(c > 0, s / np.maximum(c, 1), np.nan)


def _half_state(model):
    return {k: (v.detach().to("cpu", torch.float16) if v.is_floating_point() else v.detach().cpu().clone())
            for k, v in model.state_dict().items()}


def save_checkpoint(path, model, opt, sched, scaler, meta: dict) -> None:
    tmp = str(path) + ".tmp"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "scaler": scaler.state_dict(), "meta": meta}, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, model, opt, sched, scaler) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    sched.load_state_dict(ck["sched"])
    scaler.load_state_dict(ck["scaler"])
    return ck["meta"]


def predict_frame(ckpt_path, df: pd.DataFrame, text_col: str = "text", device=None) -> np.ndarray:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = Config(**ck["cfg"])
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = cfg.amp and device.type == "cuda"
    tok = AutoTokenizer.from_pretrained(cfg.encoder)
    tok.model_max_length = 10 ** 9
    model = BinaryEncoder(cfg.encoder, cfg.dropout).to(device)
    model.load_state_dict(ck["state"])
    pool = build_pool(tok, df[text_col].tolist(), None, cfg.max_len, cfg.max_eval_chunks, cfg.max_comments_per_author)
    return predict_pool(model, pool.eval_items(), len(df), device, amp, cfg.batch_size * 2, tok.pad_token_id)



def _make_val_split(texts, labels, ids, cfg: Config):
    idx = np.arange(len(texts))
    try:
        tr, va = train_test_split(idx, test_size=cfg.val_size, random_state=cfg.split_seed, stratify=labels)
    except ValueError:
        tr, va = train_test_split(idx, test_size=cfg.val_size, random_state=cfg.split_seed)
    pick = lambda ix: ([texts[i] for i in ix], labels[ix], [ids[i] for i in ix])
    return pick(tr), pick(va)


def _subsample(texts, labels, ids, n: int, seed: int):
    if not n or len(texts) <= n:
        return texts, labels, ids
    ix = np.sort(np.random.default_rng(seed).choice(len(texts), n, replace=False))
    return [texts[i] for i in ix], labels[ix], [ids[i] for i in ix]


def train_run(train_csv, val_csv, target: str, out_dir, cfg: Config,
              test_data=None, save_weights: bool = False, resume: bool = True) -> dict:

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(cfg.amp and dev.type == "cuda")
    t0 = time.time()

    tr_txt, tr_y, tr_ids, tr_drop = read_split(train_csv, target)
    if val_csv and Path(val_csv).exists():
        va_txt, va_y, va_ids, _ = read_split(val_csv, target)
    else:
        (tr_txt, tr_y, tr_ids), (va_txt, va_y, va_ids) = _make_val_split(tr_txt, tr_y, tr_ids, cfg)
    tr_txt, tr_y, tr_ids = _subsample(tr_txt, tr_y, tr_ids, cfg.max_authors, cfg.seed)
    va_txt, va_y, va_ids = _subsample(va_txt, va_y, va_ids, max(cfg.max_authors // 4, 50) if cfg.max_authors else 0, cfg.seed)
    if tr_drop:
        print(f"   [warn] train: {tr_drop} empty rows were discarded")

    tok = AutoTokenizer.from_pretrained(cfg.encoder)
    tok.model_max_length = 10 ** 9
    pad = tok.pad_token_id
    tr_pool = build_pool(tok, tr_txt, tr_y, cfg.max_len, cfg.max_pool_chunks, cfg.max_comments_per_author)
    va_pool = build_pool(tok, va_txt, va_y, cfg.max_len, cfg.max_eval_chunks, cfg.max_comments_per_author)
    va_items = va_pool.eval_items()

    model = BinaryEncoder(cfg.encoder, cfg.dropout, 1, cfg.grad_checkpointing).to(dev)
    nd = ("bias", "LayerNorm.weight")

    def groups(named, lr):
        named = list(named)
        return [{"params": [p for n, p in named if not any(k in n for k in nd)], "lr": lr, "weight_decay": cfg.weight_decay},
                {"params": [p for n, p in named if any(k in n for k in nd)], "lr": lr, "weight_decay": 0.0}]

    opt = torch.optim.AdamW(groups(model.encoder.named_parameters(), cfg.lr_encoder)
                            + groups(model.head.named_parameters(), cfg.lr_head))
    steps_per_epoch = max(1, -(-tr_pool.epoch_size(cfg.chunks_per_author) // cfg.batch_size))
    total = steps_per_epoch * cfg.epochs
    sched = get_linear_schedule_with_warmup(opt, int(cfg.warmup * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    n_pos = int(tr_y.sum())
    n_neg = len(tr_y) - n_pos
    pw_val = float(np.clip(n_neg / n_pos, cfg.pos_weight_min, cfg.pos_weight_max)) if n_pos else 1.0
    pw = torch.tensor(pw_val, device=dev)
    print(f"   autori train={len(tr_txt)} val={len(va_txt)} | chunk/epoca={tr_pool.epoch_size(cfg.chunks_per_author)} | pos_weight={pw_val:.2f}")

    meta = {"epoch": 0, "best_score": -1.0, "best_epoch": 0, "bad": 0, "history": [], "best_state": None}
    last = out_dir / "last.pt"
    if resume and last.exists():
        meta = load_checkpoint(last, model, opt, sched, scaler)
        print(f"   started from checkpoint: completed epoch {meta['epoch']}")

    mb = max(1, min(cfg.micro_batch, cfg.batch_size))
    for epoch in range(meta["epoch"] + 1, cfg.epochs + 1):
        if meta["bad"] >= cfg.patience:
            break
        model.train()
        rng = np.random.default_rng(cfg.seed * 1000 + epoch)
        items = tr_pool.sample_epoch(cfg.chunks_per_author, rng)
        loader = DataLoader(ChunkDataset(items), batch_size=cfg.batch_size, shuffle=False, collate_fn=make_collate(pad))
        run_loss, steps = 0.0, 0
        for inp, att, y, _ in loader:
            opt.zero_grad(set_to_none=True)
            B = len(y)
            for xi, ai, yi in zip(inp.split(mb), att.split(mb), y.split(mb)):
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                    logits = model(xi.to(dev), ai.to(dev))
                loss = F.binary_cross_entropy_with_logits(logits.float(), yi.float().to(dev), pos_weight=pw) * (len(yi) / B)
                scaler.scale(loss).backward()
                run_loss += float(loss.item())
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            steps += 1

        p_val = np.nan_to_num(predict_pool(model, va_items, len(va_txt), dev, amp, cfg.batch_size * 2, pad), nan=0.5)
        m = val_metrics(va_y, p_val)
        score = m["auc"] if not np.isnan(m["auc"]) else m["macro_f1"]  # epoca migliore = AUC di validation
        meta["history"].append({"epoch": epoch, "train_loss": run_loss / max(steps, 1), **m})
        print(f"   epoca {epoch}: loss={run_loss / max(steps, 1):.4f} val_auc={m['auc']:.3f} val_macroF1={m['macro_f1']:.3f}")
        if score > meta["best_score"]:
            meta.update(best_score=score, best_epoch=epoch, bad=0, best_state=_half_state(model))
        else:
            meta["bad"] += 1
            if meta["bad"] >= cfg.patience:
                print("   early stopping")
        meta["epoch"] = epoch
        save_checkpoint(last, model, opt, sched, scaler, meta)

    model.load_state_dict(meta["best_state"])
    p_val = np.nan_to_num(predict_pool(model, va_items, len(va_txt), dev, amp, cfg.batch_size * 2, pad), nan=0.5)
    pd.DataFrame({"id": va_ids, "prob": p_val, "label": va_y}).to_csv(out_dir / "val_probs.csv", index=False)
    val_m = val_metrics(va_y, p_val)

    test_m = None
    if test_data is not None:
        te_txt, te_y, te_ids = test_data
        te_pool = build_pool(tok, te_txt, te_y, cfg.max_len, cfg.max_eval_chunks, cfg.max_comments_per_author)
        p_te = predict_pool(model, te_pool.eval_items(), len(te_txt), dev, amp, cfg.batch_size * 2, pad)
        pd.DataFrame({"id": te_ids, "prob": p_te, "label": te_y if te_y is not None else np.nan}).to_csv(
            out_dir / "test_probs.csv", index=False)
        if te_y is not None:
            test_m = val_metrics(te_y, np.nan_to_num(p_te, nan=0.5))

    if save_weights:
        torch.save({"cfg": asdict(cfg), "target": target, "state": meta["best_state"]}, out_dir / "best.pt")
    if last.exists():
        last.unlink()

    result = {"target": target, "best_epoch": meta["best_epoch"], "val": val_m, "test": test_m,
              "history": meta["history"], "n_train_authors": len(tr_txt), "n_train_chunks_per_epoch": tr_pool.epoch_size(cfg.chunks_per_author),
              "n_val_authors": len(va_txt), "pos_weight": pw_val, "train_rows_dropped_no_text": tr_drop,
              "seconds": round(time.time() - t0, 1), "config": asdict(cfg)}
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))  # scritto per ultimo = "completato"

    del model, opt, sched, scaler, meta
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result
