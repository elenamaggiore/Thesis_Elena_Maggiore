"""Valuta checkpoint RoBERTa sul test `final` (con ensemble e bootstrap appaiato).

Un solo modello:      python scripts/evaluate.py --ckpt runs/full__noleak_full__mbti__s42 --docs-dir ...
Ensemble (seed):      python scripts/evaluate.py --ckpt run_s1 run_s2 run_s3 --docs-dir ...
Big Five vs MBTI:     python scripts/evaluate.py --ckpt <big_five...> --pair-with <mbti...> --docs-dir ...
                      (stessi autori, stesso ricampionamento: IC della differenza A - B)

Output in --out: metrics.csv (accuracy, balanced accuracy, macro-F1, F1 positiva, MCC, AUC
per task e media, con IC 95%), metrics.json e, con --pair-with, paired_difference.csv.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.documents import load_documents  # noqa: E402
from src.eval.evaluate import evaluate_checkpoints, results_table  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", nargs="+", required=True, help="cartelle dei run (uno = modello singolo, piu' = ensemble)")
    ap.add_argument("--pair-with", nargs="+", default=None, help="secondo ensemble (altra famiglia) sugli stessi autori")
    ap.add_argument("--docs-dir", required=True, help="cartella dei documents_<livello>_final.jsonl.gz")
    ap.add_argument("--out", default="results/eval")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--recompute", action="store_true", help="rifa le predizioni anche se sono in cache")
    args = ap.parse_args()

    level = json.loads((Path(args.ckpt[0]) / "config.json").read_text())["level"]
    test = load_documents(Path(args.docs_dir) / f"documents_{level}_final.jsonl.gz")
    res = evaluate_checkpoints(args.ckpt, test, pair_with=args.pair_with, B=args.bootstrap,
                               boot_seed=args.boot_seed, thr=args.thr, batch_size=args.batch_size,
                               recompute=args.recompute)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tab = results_table(res["a"])
    tab.to_csv(out / "metrics.csv", index=False)
    print(f"Config: {res['config']} | checkpoint: {res['n_checkpoints']} | autori: {res['a']['n_authors']}")
    print(tab.round(3).to_string(index=False))
    if "b" in res:
        tab_b = results_table(res["b"])
        tab_b.to_csv(out / "metrics_pair_with.csv", index=False)
        d = pd.DataFrame([{"metric": m, **v} for m, v in res["paired_difference"].items()])
        d.to_csv(out / "paired_difference.csv", index=False)
        print("\nDifferenza A - B (metriche medie sui task, IC 95% appaiato):")
        print(d.round(3).to_string(index=False))
    slim = {k: v for k, v in res.items() if k not in ("a", "b")}
    for k in ("a", "b"):
        if k in res:
            slim[k] = {kk: vv for kk, vv in res[k].items() if not kk.startswith("_")}
    (out / "metrics.json").write_text(json.dumps(slim, indent=2, default=float))
    print(f"\nRisultati in {out}")


if __name__ == "__main__":
    main()
