"""
Trains a RoBERTa model for each combination of masking/strategy/dataset/target, for a total of
180 RoBERTa models. Models already trained are skipped.

  python run_training.py --data_root Datasets_ready_to_train --out_root Models_s11 --test_root Datasets_test --dry-run
  python run_training.py --data_root ... --out_root Models_s11 --test_root ...
  python run_training.py ... --seed 12 --out_root Models_s12
  python run_training.py ... --maskings none --datasets big_five --targets openness --strategies no_oversampling

"""
from __future__ import annotations

import argparse
import traceback
from dataclasses import fields
from pathlib import Path
import pandas as pd
import torch

from src.models.chunks import read_split
from src.models.roberta import Config, train_run

MASKINGS = ["none", "limited", "full", "llm"]
TARGETS = {
    "mbti": ["introverted", "intuitive", "thinking", "perceiving"],
    "big_five": ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"],
}
STRATEGIES = ["paper_leaky", "noleak_full", "noleak_minimal", "no_oversampling", "undersampling"]


def find_test_file(test_root: Path, masking: str = None):
    p = test_root / "test_text.csv"
    return p if p.exists() else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--test_root", default=None, help="Final test set root")
    ap.add_argument("--maskings", nargs="*", default=MASKINGS)
    ap.add_argument("--datasets", nargs="*", default=list(TARGETS))
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--strategies", nargs="*", default=STRATEGIES)
    ap.add_argument("--save_weights", action="store_true", help="saves best weights for each model")
    ap.add_argument("--no_resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    for f in fields(Config):
        if f.type in ("int", "float", "str"):
            ap.add_argument(f"--{f.name}", type={"int": int, "float": float, "str": str}[f.type], default=None)
        else:
            ap.add_argument(f"--{f.name}", type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
    args = ap.parse_args()

    cfg = Config()
    for f in fields(Config):
        v = getattr(args, f.name, None)
        if v is not None:
            setattr(cfg, f.name, v)

    data_root, out_root = Path(args.data_root), Path(args.out_root)
    test_root = Path(args.test_root) if args.test_root else None

    jobs, missing = [], []
    for mk in args.maskings:
        for ds in args.datasets:
            for tg in TARGETS[ds]:
                if args.targets and tg not in args.targets:
                    continue
                for st in args.strategies:
                    d = data_root / mk / ds / tg / st
                    (jobs if (d / "train_text.csv").exists() else missing).append((mk, ds, tg, st))
    print(f"Training: {len(jobs)} | missing: {len(missing)}")
    for m in missing[:10]:
        print("  mancante:", "/".join(m))
    if args.dry_run:
        return

    print(f"Device: {'cuda' if torch.cuda.is_available() else 'CPU (slow)'}\nConfig: {cfg}")
    out_root.mkdir(parents=True, exist_ok=True)
    summary = out_root / "training_summary.csv"
    test_cache = {}

    for i, (mk, ds, tg, st) in enumerate(jobs, 1):
        out_dir = out_root / mk / ds / tg / st
        tag = f"[{i}/{len(jobs)}] {mk}/{ds}/{tg}/{st}"
        if (out_dir / "metrics.json").exists():
            print(f"{tag}: skipped (already present)")
            continue
        print(f"\n{tag}")

        test_data = None
        if test_root is not None:
            if mk not in test_cache:
                test_cache[mk] = find_test_file(test_root, mk)
                if test_cache[mk] is None:
                    print(f"   [warn] no test set for '{mk}' in {test_root}")
            if test_cache[mk] is not None:
                try:
                    txt, y, ids, _ = read_split(test_cache[mk], tg, allow_generic_label=False)
                    test_data = (txt, y, ids)
                except KeyError as e:
                    print(f"   [warn] invalid test set for {tg}: {e}")

        d = data_root / mk / ds / tg / st
        try:
            r = train_run(d / "train_text.csv", d / "val_text.csv", tg, out_dir, cfg,
                          test_data=test_data, save_weights=args.save_weights, resume=not args.no_resume)
        except Exception:  # un errore (es. OOM) non ferma gli altri modelli; rilanciando si riprende da last.pt
            print(f"Error on {tag}:\n{traceback.format_exc()}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        row = {"masking": mk, "dataset": ds, "target": tg, "strategy": st, "seed": cfg.seed,
               "best_epoch": r["best_epoch"], "n_train_authors": r["n_train_authors"],
               "pos_weight": r["pos_weight"], "seconds": r["seconds"],
               **{f"val_{k}": v for k, v in r["val"].items()}}
        if r["test"]:
            row.update({f"test_{k}": v for k, v in r["test"].items()})
        pd.DataFrame([row]).to_csv(summary, mode="a", header=not summary.exists(), index=False)

    print("\nDone. Summary:", summary)


if __name__ == "__main__":
    main()
