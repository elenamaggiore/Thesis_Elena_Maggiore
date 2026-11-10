"""Applica le strategie di sampling e stampa i conteggi per classe a ogni passo.

Serve a VERIFICARE le strategie (soprattutto che quella senza leakage abbia
val_rows_with_source_in_train = 0) prima di addestrare i modelli. Per default non
calcola le feature (veloce); con --features calcola anche Word2Vec + SMOTE.
I sintetici vengono letti dalla cache creata da generate_synthetic.py.

Per default scorre TUTTI i livelli di masking (none, limited, full, llm), i dataset di
training (mbti, big_five), i loro target e le cinque strategie.

Esempio:
  python scripts/run_sampling_report.py --docs-dir "C:\\...\\Documents" \
      --cache "C:\\...\\Documents\\gpt2_cache.sqlite" --out-csv report.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.documents import load_documents  # noqa: E402
from src.data.targets import TARGETS  # noqa: E402
from src.sampling.gpt2_aug import CacheOnlyGenerator, SyntheticCache, make_signature  # noqa: E402
from src.sampling.prepare import prepare_target  # noqa: E402
from src.sampling.strategies import STRATEGIES  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs-dir", required=True)
    ap.add_argument("--levels", nargs="+", default=["none", "limited", "full", "llm"])
    ap.add_argument("--datasets", nargs="+", default=["mbti", "big_five"], choices=["mbti", "big_five"])
    ap.add_argument("--cache", help="cache SQLite dei sintetici (serve alle strategie con GPT-2)")
    ap.add_argument("--targets", nargs="+", default=None, help="solo alcuni target (default: tutti)")
    ap.add_argument("--strategies", nargs="+", default=list(STRATEGIES), choices=list(STRATEGIES))
    ap.add_argument("--features", action="store_true", help="calcola anche Word2Vec e SMOTE")
    ap.add_argument("--max-chunks", type=int, default=8, help="deve coincidere con generate_synthetic.py")
    ap.add_argument("--seed", type=int, default=42, help="seed di split/sampling")
    ap.add_argument("--gpt2-model", default="gpt2", help="deve coincidere con generate_synthetic.py")
    ap.add_argument("--gen-seed", type=int, default=42, help="il --seed usato in generate_synthetic.py")
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    cache = SyntheticCache(args.cache) if args.cache else None
    reader = CacheOnlyGenerator(make_signature(args.gpt2_model, seed=args.gen_seed))
    rows = []
    for level in args.levels:
        for dataset in args.datasets:
            path = Path(args.docs_dir) / f"documents_{level}_{dataset}.jsonl.gz"
            if not path.exists():
                print(f"[{level}/{dataset}] documenti non trovati: {path}")
                continue
            docs = load_documents(path)
            for target in (args.targets or TARGETS[dataset]):
                for name in args.strategies:
                    try:
                        p = prepare_target(docs, target, name, generator=reader, cache=cache, seed=args.seed,
                                           max_chunks=args.max_chunks or None, features=args.features)
                    except RuntimeError as e:
                        print(f"[{level}/{dataset}/{target}/{name}] {e}")
                        continue
                    r = p.report
                    rows.append({
                        "level": level, "dataset": dataset, "target": target, "strategy": name,
                        "train_pos": r["train_text_level"]["pos"], "train_neg": r["train_text_level"]["neg"],
                        "val_pos": r["val"]["pos"], "val_neg": r["val"]["neg"],
                        "synthetic_added": r.get("synthetic_added", 0),
                        "val_synthetic_rows": r["val_synthetic_rows"],
                        "val_rows_with_source_in_train": r["val_rows_with_source_in_train"],
                        "train_vectors_pos": r.get("train_vectors", {}).get("pos"),
                        "train_vectors_neg": r.get("train_vectors", {}).get("neg"),
                    })
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    if args.out_csv:
        out.to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    main()
