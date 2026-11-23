"""
Testing RoBERTa on the final dataset, for each combination of masking and sampling strategy.
The four MBTI models and the five Big Five models are combined. Confidence Intervals at 95%
are based on linked bootstrap between MBTI and Big Five datasets.

Ensemble option (many runs with different seeds are averaged by author) can be activated, if
models with same configuration but different seeds have been trained.

python evaluate.py --ckpt Models_s11 Models_s12 Models_s13 --out test_results --n_boot 2000

"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.roberta import METRICS, val_metrics

warnings.filterwarnings("ignore", category=RuntimeWarning)

MASKINGS = ["none", "limited", "full", "llm"]
STRATEGIES = ["paper_leaky", "noleak_full", "noleak_minimal", "no_oversampling", "undersampling"]
TARGETS = {
    "mbti": ["introverted", "intuitive", "thinking", "perceiving"],
    "big_five": ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"],
}
COLS = METRICS + ["exact_match"]


def load_dataset(roots, mk, ds, st):
    merged, n_seeds = None, []
    for tg in TARGETS[ds]:
        frames = [pd.read_csv(Path(r) / mk / ds / tg / st / "test_probs.csv", dtype={"id": str})
                  for r in roots if (Path(r) / mk / ds / tg / st / "test_probs.csv").exists()]
        if not frames:
            return None, 0
        n_seeds.append(len(frames))
        g = pd.concat(frames).groupby("id", sort=False).agg(p=("prob", "mean"), y=("label", "first"))
        g.columns = [f"p_{tg}", f"y_{tg}"]
        merged = g if merged is None else merged.join(g, how="inner")
    return merged, min(n_seeds)


def arrays(D, ds):
    y = np.stack([D[f"y_{t}"].to_numpy().astype(int) for t in TARGETS[ds]], 1)
    p = np.stack([D[f"p_{t}"].to_numpy() for t in TARGETS[ds]], 1)
    return y, p


def combined(y, p):
    M = np.array([[val_metrics(y[:, j], p[:, j])[k] for k in METRICS] for j in range(y.shape[1])])
    exact = float(((p >= 0.5).astype(int) == y).all(1).mean())
    return np.append(np.nanmean(M, 0), exact), M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--out", default="test_results")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    per_target, summary = [], []
    for mk in MASKINGS:
        for st in STRATEGIES:
            Dm, sm = load_dataset(args.ckpt, mk, "mbti", st)
            Db, sb = load_dataset(args.ckpt, mk, "big_five", st)
            if Dm is None or Db is None:
                continue
            common = Dm.index.intersection(Db.index)
            Dm, Db = Dm.loc[common], Db.loc[common]
            ym, pm = arrays(Dm, "mbti")
            yb, pb = arrays(Db, "big_five")
            A = len(common)
            if A == 0:
                continue

            point, tabs = {}, {}
            for name, (y, p) in {"mbti": (ym, pm), "big_five": (yb, pb)}.items():
                point[name], tabs[name] = combined(y, p)
                for j, tg in enumerate(TARGETS[name]):
                    per_target.append({"masking": mk, "strategy": st, "dataset": name, "target": tg,
                                       "n_seeds": sm if name == "mbti" else sb,
                                       **val_metrics(y[:, j], p[:, j])})

            boot_m, boot_b = [], []
            for _ in range(args.n_boot):  # linked bootstrap
                idx = rng.integers(0, A, A)
                boot_m.append(combined(ym[idx], pm[idx])[0])
                boot_b.append(combined(yb[idx], pb[idx])[0])
            boot_m, boot_b = np.array(boot_m), np.array(boot_b)
            blocks = {"mbti": (point["mbti"], boot_m), "big_five": (point["big_five"], boot_b),
                      "mbti_minus_big_five": (point["mbti"] - point["big_five"], boot_m - boot_b)}
            for name, (pt, bt) in blocks.items():
                lo, hi = np.nanpercentile(bt, [2.5, 97.5], axis=0)
                for k, col in enumerate(COLS):
                    summary.append({"masking": mk, "strategy": st, "dataset": name, "metric": col,
                                    "value": pt[k], "ci95_low": lo[k], "ci95_high": hi[k],
                                    "n_authors": A, "n_seeds": min(sm, sb), "n_boot": args.n_boot})

    if not summary:
        print("test_probs.csv not found: did you put --test_root in run_training.py?")
        return
    pd.DataFrame(per_target).to_csv(out / "test_per_target.csv", index=False)
    S = pd.DataFrame(summary)
    S.to_csv(out / "test_combined_with_ci.csv", index=False)
    piv = S[S.metric == "macro_f1"].pivot_table(index=["masking", "strategy"], columns="dataset", values="value")
    print("Macro-F1 averaged on targets:\n", piv.round(3).to_string())
    print("\nSaved in", out)


if __name__ == "__main__":
    main()
