
"""Apply sampling strategies using existing synthetic CSVs.

For each masking level, dataset, target and strategy, saves:
  - train_text.csv       Text-level training data
  - val_text.csv         Text-level validation data
  - train_features.csv   Word2Vec training vectors (SMOTE if applicable)
  - val_features.csv     Word2Vec validation vectors (never SMOTE)
  - report.csv           Sampling and leakage diagnostics

GPT-2 synthetic documents are loaded from existing CSV files.
No text generation or SQLite cache access is performed.
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
from src.sampling.prepare import prepare_target
from src.sampling.strategies import STRATEGIES


def safe_name(value):
    """Make a string safe for use as a filename."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def load_synthetic_csv(synthetic_dir, level, dataset, target, kind):
    """Load previously generated synthetic documents from CSV."""

    path = (
        Path(synthetic_dir)
        / level
        / dataset
        / f"{target}_{kind}.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Synthetic CSV not found: {path}"
        )

    syn = pd.read_csv(path)

    required = {
        "author",
        "source_author",
        "text",
        target,
        "is_synthetic",
        "synthetic_target",
        "input_strategy",
        "masking_level",
        "dataset",
    }

    missing = required - set(syn.columns)

    if missing:
        raise ValueError(
            f"Missing columns in {path}: {sorted(missing)}"
        )

    # Keep only rows matching the requested experiment.
    syn = syn[
        (syn["synthetic_target"] == target)
        & (syn["input_strategy"] == kind)
        & (syn["masking_level"] == level)
        & (syn["dataset"] == dataset)
    ].copy()

    if syn.empty:
        raise ValueError(
            f"No synthetic documents found in {path} "
            f"for target={target}, input={kind}, level={level}"
        )

    # Ensure synthetic rows are correctly marked.
    syn["is_synthetic"] = True

    # Check that source authors are available.
    if syn["source_author"].isna().any():
        raise ValueError(
            f"Missing source_author values in {path}"
        )

    return syn.reset_index(drop=True)


def save_text_csv(df, path):
    """Save text-level data, serializing list-valued columns."""

    out = df.copy()

    for col in out.columns:
        if out[col].map(
            lambda x: isinstance(x, (list, dict))
        ).any():
            out[col] = out[col].apply(
                lambda x: json.dumps(x, ensure_ascii=False)
                if isinstance(x, (list, dict))
                else x
            )

    out.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


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

    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


def build_report_row(level, dataset, target, strategy, report):
    """Build one row of the global sampling report."""

    return {
        "level": level,
        "dataset": dataset,
        "target": target,
        "strategy": strategy,

        "train_pos": report["train_text_level"]["pos"],
        "train_neg": report["train_text_level"]["neg"],

        "val_pos": report["val"]["pos"],
        "val_neg": report["val"]["neg"],

        "synthetic_added": report.get(
            "synthetic_added", 0
        ),

        "val_synthetic_rows": report[
            "val_synthetic_rows"
        ],

        "val_rows_with_source_in_train": report[
            "val_rows_with_source_in_train"
        ],

        "train_vectors_pos": report.get(
            "train_vectors", {}
        ).get("pos"),

        "train_vectors_neg": report.get(
            "train_vectors", {}
        ).get("neg"),

        "smote": json.dumps(
            report.get("smote", None),
            ensure_ascii=False,
        ),
    }


def main():

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input and output directories.
    ap.add_argument(
        "--docs-dir",
        required=True,
        help="Directory containing the built document datasets",
    )

    ap.add_argument(
        "--synthetic-dir",
        required=True,
        help="Directory containing the existing synthetic CSV files",
    )

    ap.add_argument(
        "--out-dir",
        default="Datasets_ready_to_train",
        help="Directory where sampled datasets will be saved",
    )

    # Experiment configuration.
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

    ap.add_argument(
        "--targets",
        nargs="+",
        default=None,
        help="Targets to process; by default all targets are used",
    )

    ap.add_argument(
        "--strategies",
        nargs="+",
        default=list(STRATEGIES),
        choices=list(STRATEGIES),
    )

    # Sampling configuration.
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    ap.add_argument(
        "--val-size",
        type=float,
        default=0.2,
    )

    # Feature configuration.
    ap.add_argument(
        "--no-features",
        action="store_true",
        help="Skip Word2Vec and SMOTE feature computation",
    )

    # Output configuration.
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )

    args = ap.parse_args()

    docs_dir = Path(args.docs_dir)
    synthetic_dir = Path(args.synthetic_dir)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_rows = []

    for level in args.levels:

        for dataset in args.datasets:

            docs_path = (
                docs_dir
                / level
                / f"documents_{dataset}.jsonl.gz"
            )

            if not docs_path.exists():
                print(
                    f"[{level}/{dataset}] "
                    f"Documents not found: {docs_path}",
                    flush=True,
                )
                continue

            print(
                f"\n{'=' * 70}\n"
                f"LEVEL: {level} | DATASET: {dataset}\n"
                f"{'=' * 70}",
                flush=True,
            )

            docs = load_documents(docs_path)

            targets = args.targets or TARGETS[dataset]

            for target in targets:

                if target not in docs.columns:
                    print(
                        f"[{level}/{dataset}/{target}] "
                        "Target not found in documents. Skipping.",
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
                        "report.csv",
                    ]

                    if not args.no_features:
                        filenames.extend([
                            "train_features.csv",
                            "val_features.csv",
                        ])

                    if (
                        not args.overwrite
                        and all(
                            (output_path / name).exists()
                            for name in filenames
                        )
                    ):
                        print(
                            f"[{level}/{dataset}/{target}/{strategy}] "
                            "Already exported; skipping.",
                            flush=True,
                        )
                        continue

                    print(
                        f"\n[{level}/{dataset}/{target}/{strategy}] "
                        "Preparing...",
                        flush=True,
                    )

                    try:

                        synthetic_docs = None

                        # Load the existing synthetic CSV only for
                        # strategies that use GPT-2 augmentation.
                        if STRATEGIES[strategy].gpt2:

                            kind = STRATEGIES[
                                strategy
                            ].preprocessing

                            synthetic_docs = load_synthetic_csv(
                                synthetic_dir=synthetic_dir,
                                level=level,
                                dataset=dataset,
                                target=target,
                                kind=kind,
                            )

                            print(
                                f"  Loaded {len(synthetic_docs)} "
                                f"synthetic documents "
                                f"(input={kind}).",
                                flush=True,
                            )

                        # Prepare train/validation datasets.
                        # No GPT-2 generation and no SQLite cache.
                        prepared = prepare_target(
                            docs=docs,
                            target=target,
                            strategy=strategy,
                            synthetic_docs=synthetic_docs,
                            seed=args.seed,
                            val_size=args.val_size,
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

                        # 2. Save Word2Vec features.
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

                        # 3. Save individual strategy report.
                        report = prepared.report

                        report_row = build_report_row(
                            level,
                            dataset,
                            target,
                            strategy,
                            report,
                        )

                        pd.DataFrame(
                            [report_row]
                        ).to_csv(
                            output_path / "report.csv",
                            index=False,
                            encoding="utf-8-sig",
                        )

                        report_rows.append(report_row)

                        print(
                            f"[{level}/{dataset}/{target}/{strategy}] "
                            f"Train: {report['train_text_level']} | "
                            f"Validation: {report['val']} | "
                            f"Leakage: "
                            f"{report['val_rows_with_source_in_train']}",
                            flush=True,
                        )

                        print(
                            f"  Saved in: {output_path}",
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

    # 4. Save global sampling report.
    if report_rows:

        report = pd.DataFrame(report_rows)

        report_path = (
            out_dir / "sampling_report.csv"
        )

        report.to_csv(
            report_path,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            "\n" + "=" * 70
        )

        print("SAMPLING COMPLETED")

        print(
            "=" * 70
        )

        print(
            report.to_string(index=False)
        )

        print(
            f"\nGlobal report saved to: {report_path}"
        )

    else:

        print(
            "\nNo datasets were processed."
        )


if __name__ == "__main__":
    main()
