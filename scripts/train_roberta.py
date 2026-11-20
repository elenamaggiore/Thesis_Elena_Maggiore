"""Addestra UN run RoBERTa su uno dei 20 dataset (livello di masking x strategia).

Esempio:
  python scripts/train_roberta.py --docs-dir "$D/Documents" --runs-dir "$D/runs" \
      --cache "$D/Documents/gpt2_cache.sqlite" --level full --strategy noleak_full --dataset mbti --seed 42
Pilota veloce (pochi autori, 1 epoca):  aggiungi  --limit-authors 200 --epochs 1
Tutti i parametri del modello (--encoder, --dropout, --lr-encoder, --lr-head, --batch-size,
--epochs, --patience, ...) hanno i default della specifica; vedi --help.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.roberta import DEFAULTS  # noqa: E402
from src.models.roberta_run import run_one  # noqa: E402
from src.sampling.multitask import MASKING_LEVELS  # noqa: E402
from src.sampling.strategies import STRATEGIES  # noqa: E402


def add_cfg_args(ap):
    """Un argomento per ogni parametro in DEFAULTS (nome con trattini)."""
    for k, v in DEFAULTS.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            ap.add_argument("--no-" + k.replace("_", "-"), dest=k, action="store_false", default=v,
                            help=f"disattiva {k}")
        else:
            ap.add_argument(flag, dest=k, type=type(v), default=v, help=f"default {v}")


def cfg_from_args(args) -> dict:
    return {k: getattr(args, k) for k in DEFAULTS}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs-dir", required=True)
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--cache", help="cache dei sintetici (serve alle strategie con GPT-2)")
    ap.add_argument("--level", required=True, choices=list(MASKING_LEVELS))
    ap.add_argument("--strategy", required=True, choices=list(STRATEGIES))
    ap.add_argument("--dataset", required=True, choices=["mbti", "big_five"])
    ap.add_argument("--gpt2-model", default="gpt2", help="deve coincidere con generate_synthetic.py")
    ap.add_argument("--gen-seed", type=int, default=42, help="il --seed usato in generate_synthetic.py")
    ap.add_argument("--max-chunks", type=int, default=8, help="deve coincidere con generate_synthetic.py")
    ap.add_argument("--limit-authors", type=int, default=None, help="pilota: usa solo N autori")
    add_cfg_args(ap)
    args = ap.parse_args()
    run_one(args.level, args.strategy, args.dataset, cfg_from_args(args), docs_dir=args.docs_dir,
            runs_dir=args.runs_dir, cache_path=args.cache, gpt2_model=args.gpt2_model,
            gen_seed=args.gen_seed, max_chunks=args.max_chunks or None, limit_authors=args.limit_authors)


if __name__ == "__main__":
    main()
