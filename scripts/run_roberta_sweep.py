"""RoBERTa sui 20 dataset (4 livelli di masking x 5 strategie), per le due famiglie di target.

Per ogni (livello, strategia) si addestra un modello MBTI (4 uscite) e uno Big Five (5 uscite),
per ogni seed. Un run completato (done.json) viene saltato: se si interrompe, si rilancia lo
stesso comando. Un run che fallisce (es. sintetici mancanti in cache) viene segnalato e
si passa al successivo.

  python scripts/run_roberta_sweep.py --list ...        # elenca i run e quali sono gia' fatti
  python scripts/run_roberta_sweep.py --docs-dir ... --runs-dir ... --cache ... --seeds 42 43 44
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_roberta import add_cfg_args, cfg_from_args  # noqa: E402
from src.data.documents import load_documents  # noqa: E402
from src.models.roberta_run import docs_path, run_name, run_one  # noqa: E402
from src.sampling.multitask import MASKING_LEVELS  # noqa: E402
from src.sampling.strategies import STRATEGIES  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs-dir", required=True)
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--cache")
    ap.add_argument("--levels", nargs="+", default=list(MASKING_LEVELS), choices=list(MASKING_LEVELS))
    ap.add_argument("--strategies", nargs="+", default=list(STRATEGIES), choices=list(STRATEGIES))
    ap.add_argument("--datasets", nargs="+", default=["mbti", "big_five"], choices=["mbti", "big_five"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--gpt2-model", default="gpt2")
    ap.add_argument("--gen-seed", type=int, default=42)
    ap.add_argument("--max-chunks", type=int, default=8)
    ap.add_argument("--limit-authors", type=int, default=None)
    ap.add_argument("--list", action="store_true", help="elenca i run senza eseguirli")
    add_cfg_args(ap)
    ap.set_defaults(seed=None)                         # il seed arriva da --seeds
    args = ap.parse_args()
    cfg = cfg_from_args(args)

    plan = [(lv, st, ds, sd) for lv in args.levels for ds in args.datasets
            for st in args.strategies for sd in args.seeds]
    runs = Path(args.runs_dir)
    if args.list:
        for lv, st, ds, sd in plan:
            done = (runs / run_name(lv, st, ds, sd, bool(args.limit_authors)) / "done.json").exists()
            print(f"{'FATTO ' if done else 'DA FARE'}  {run_name(lv, st, ds, sd)}")
        print(f"\n{len(plan)} run ({len(args.levels)} livelli x {len(args.strategies)} strategie x "
              f"{len(args.datasets)} famiglie x {len(args.seeds)} seed)")
        return

    ok = skipped = failed = 0
    for lv in args.levels:
        for ds in args.datasets:
            path = docs_path(args.docs_dir, lv, ds)
            if not path.exists():
                print(f"[{lv}/{ds}] documenti non trovati: {path}")
                failed += len(args.strategies) * len(args.seeds)
                continue
            docs = load_documents(path)                # un solo caricamento per (livello, famiglia)
            for st in args.strategies:
                for sd in args.seeds:
                    try:
                        r = run_one(lv, st, ds, {**cfg, "seed": sd}, docs_dir=args.docs_dir, runs_dir=args.runs_dir,
                                    cache_path=args.cache, gpt2_model=args.gpt2_model, gen_seed=args.gen_seed,
                                    max_chunks=args.max_chunks or None, limit_authors=args.limit_authors, docs=docs)
                        skipped += r is None
                        ok += r is not None
                    except Exception as e:             # un run fallito non ferma lo sweep
                        failed += 1
                        print(f"[{run_name(lv, st, ds, sd)}] FALLITO: {type(e).__name__}: {e}")
    print(f"\nSweep finito: {ok} completati, {skipped} gia' fatti, {failed} falliti")


if __name__ == "__main__":
    main()
