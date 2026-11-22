
"""Apply all sampling strategies and export text and Word2Vec datasets to CSV.

For each masking level, dataset, target and strategy, saves:
  - train_text.csv       Text-level training data
  - val_text.csv         Text-level validation data
  - train_features.csv   Word2Vec training vectors (SMOTE if applicable)
  - val_features.csv     Word2Vec validation vectors (never SMOTE)
  - report.csv           Sampling and leakage diagnostics

GPT-2 synthetic documents are read from the existing SQLite cache.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.documents import load_documents
from src.data.targets import TARGETS
from src.sampling.gpt2_aug import (
    CacheOnlyGenerator,
    SyntheticCache,
    make_signature,
)
from src.sampling.prepare import prepare_target
from src.sampling.strategies import STRATEGIES


def safe_name(value):
    """Make a string safe for use as a filename."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def save_text_csv(df, path):
    """Save text-level data, serializing list-valued columns."""
    out = df.copy()

    for col in out.columns:
        if out[col].map(lambda x: isinstance(x, (list, dict))).any():
            out[col] = out[col].apply(
                lambda x: json.dumps(x, ensure_ascii=False)
                if isinstance(x, (list, dict))
                else x
            )

    out.to_csv(path, index=False, encoding="utf-8-sig")


def save_feature_csv(X, y, path):
    """Save Word2Vec vectors and their labels."""
    if X is None or y is None:
        return

    X = np.asarray(X)
    y = np.asarray(y).astype(int)

    if len(X) != len(y):
        raise ValueError(
            f"Feature/label length mismatch: {len(X)} != {len(y)}"
        )

    columns = [f"w2v_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=columns)
    df.insert(0, "y", y)

    df.to_csv(path, index=False, encoding="utf-8-sig")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    ap.add_argument("--docs-dir", required=True)
    ap.add_argument("--out-dir", default="Datasets_processed")
    ap.add_argument("--cache", default=None)

    ap.add_argument(
        "--levels",
        nargs="+",
        default=["none", "limited", "full", "llm"],
    )
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=["mbti", "big_five"],
        choices=["mbti", "big_five"],
    )
    ap.add_argument("--targets", nargs="+", default=None)

    ap.add_argument(
        "--strategies",
        nargs="+",
        default=list(STRATEGIES),
        choices=list(STRATEGIES),
    )

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-size", type=float, default=0.2)

    ap.add_argument("--gpt2-model", default="gpt2")
    ap.add_argument("--gen-seed", type=int, default=42)
    ap.add_argument("--gpt2-multiplier", type=float, default=1.0)
    ap.add_argument("--chunk-size", type=int, default=5)
    ap.add_argument("--max-chunks", type=int, default=8)

    ap.add_argument(
        "--no-features",
        action="store_true",
        help="Skip Word2Vec and SMOTE feature computation",
    )

    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )

    args = ap.parse_args()

    docs_dir = Path(args.docs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # A cache is necessary whenever a GPT-2 strategy is selected.
    needs_gpt2 = any(
        STRATEGIES[name].gpt2 for name in args.strategies
    )

    if needs_gpt2 and not args.cache:
        ap.error(
            "GPT-2 strategies require --cache containing previously "
            "generated synthetic documents."
        )

    cache = SyntheticCache(args.cache) if args.cache else None

    generator = (
        CacheOnlyGenerator(
            make_signature(args.gpt2_model, seed=args.gen_seed)
        )
        if needs_gpt2
        else None
    )

    report_rows = []

    try:
        for level in args.levels:
            for dataset in args.datasets:

                docs_path = (
                    docs_dir
                    / level
                    / f"documents_{dataset}.jsonl.gz"
                )

                if not docs_path.exists():
                    print(
                        f"[{level}/{dataset}] Documents not found: "
                        f"{docs_path}",
                        flush=True,
                    )
                    continue

                docs = load_documents(docs_path)

                targets = args.targets or TARGETS[dataset]

                for target in targets:

                    if target not in docs.columns:
                        print(
                            f"[{level}/{dataset}/{target}] "
                            "Target not found",
                            flush=True,
                        )
                        continue

                    for strategy in args.strategies:

                        output_path = (
                            out_dir
                            / level
                            / dataset
                            / safe_name(target)
                            / strategy
                        )
                        output_path.mkdir(
                            parents=True,
                            exist_ok=True,
                        )

                        filenames = [
                            "train_text.csv",
                            "val_text.csv",
                            "train_features.csv",
                            "val_features.csv",
                        ]

                        if (
                            not args.overwrite
                            and all(
                                (output_path / name).exists()
                                for name in filenames
                            )
                        ):
                            print(
                                f"[{level}/{dataset}/{target}/{strategy}] "
                                "already exported; skipping",
                                flush=True,
                            )
                            continue

                        print(
                            f"\n[{level}/{dataset}/{target}/{strategy}] "
                            "Preparing...",
                            flush=True,
                        )

                        try:
                            prepared = prepare_target(
                                docs,
                                target,
                                strategy,
                                generator=generator,
                                cache=cache,
                                seed=args.seed,
                                val_size=args.val_size,
                                gpt2_multiplier=args.gpt2_multiplier,
                                chunk_size=args.chunk_size,
                                max_chunks=args.max_chunks or None,
                                features=not args.no_features,
                            )

                            # 1. Save text-level datasets.
                            save_text_csv(
                                prepared.train,
                                output_path / "train_text.csv",
                            )
                            save_text_csv(
                                prepared.val,
                                output_path / "val_text.csv",
                            )

                            # 2. Save Word2Vec vectors.
                            if not args.no_features:
                                save_feature_csv(
                                    prepared.X_train,
                                    prepared.y_train_vec,
                                    output_path / "train_features.csv",
                                )
                                save_feature_csv(
                                    prepared.X_val,
                                    prepared.y_val_vec,
                                    output_path / "val_features.csv",
                                )

                            # 3. Save strategy diagnostics.
                            r = prepared.report

                            report_row = {
                                "level": level,
                                "dataset": dataset,
                                "target": target,
                                "strategy": strategy,
                                "train_pos": r["train_text_level"]["pos"],
                                "train_neg": r["train_text_level"]["neg"],
                                "val_pos": r["val"]["pos"],
                                "val_neg": r["val"]["neg"],
                                "synthetic_added": r.get(
                                    "synthetic_added", 0
                                ),
                                "val_synthetic_rows": r[
                                    "val_synthetic_rows"
                                ],
                                "val_rows_with_source_in_train": r[
                                    "val_rows_with_source_in_train"
                                ],
                                "train_vectors_pos": r.get(
                                    "train_vectors", {}
                                ).get("pos"),
                                "train_vectors_neg": r.get(
                                    "train_vectors", {}
                                ).get("neg"),
                                "smote": json.dumps(
                                    r.get("smote", None),
                                    ensure_ascii=False,
                                ),
                            }

                            report_rows.append(report_row)

                            print(
                                f"[{level}/{dataset}/{target}/{strategy}] "
                                f"Train: {r['train_text_level']} | "
                                f"Validation: {r['val']} | "
                                f"Leakage: "
                                f"{r['val_rows_with_source_in_train']}",
                                flush=True,
                            )

                            print(
                                f"  -> Saved in {output_path}",
                                flush=True,
                            )

                        except RuntimeError as e:
                            print(
                                f"[{level}/{dataset}/{target}/{strategy}] "
                                f"ERROR: {e}",
                                flush=True,
                            )

                        except Exception as e:
                            print(
                                f"[{level}/{dataset}/{target}/{strategy}] "
                                f"UNEXPECTED ERROR: {e}",
                                flush=True,
                            )
                            raise

        # 4. Export global report.
        if report_rows:
            report = pd.DataFrame(report_rows)
            report_path = out_dir / "sampling_report.csv"
            report.to_csv(
                report_path,
                index=False,
                encoding="utf-8-sig",
            )

            print("\n" + "=" * 70)
            print("SAMPLING COMPLETED")
            print("=" * 70)
            print(report.to_string(index=False))
            print(f"\nReport saved to: {report_path}")

        else:
            print("No datasets were processed.")

    finally:
        if cache is not None:
            cache.close()


if __name__ == "__main__":
    main()
