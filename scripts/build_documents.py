"""
Phase 1: build one document per author (with binary targets)
from masked comments.

MBTI and Big Five have a separate dataset for each masking level.
The Final Dataset is processed only once from the original,
unmasked dataset.

Output:
  <out-dir>/<level>/documents_<dataset>.jsonl.gz
  <out-dir>/final/documents_final.jsonl.gz
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.documents import build_documents, save_documents  # noqa: E402
from src.masking.pipeline import DATASETS, load_author_types  # noqa: E402


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--masked-dir", required=True)
    ap.add_argument("--final-dir", required=True)
    ap.add_argument("--profiles", required=True)
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--levels", nargs="+", default=["none", "limited", "full", "llm"])
    ap.add_argument("--datasets", nargs="+", default=["mbti", "big_five", "final"],
                    choices=["mbti", "big_five", "final"])

    ap.add_argument("--max-comments", type=int, default=200)
    ap.add_argument("--bigfive-cutoff", type=float, default=50.0)
    ap.add_argument("--bigfive-cutoffs-json", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")

    args = ap.parse_args()

    cutoff = args.bigfive_cutoff
    if args.bigfive_cutoffs_json:
        cutoff = json.loads(
            Path(args.bigfive_cutoffs_json).read_text(encoding="utf-8")
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    profiles = Path(args.profiles)

    training_datasets = [
        name for name in args.datasets
        if name in ("mbti", "big_five")
    ]

    for level in args.levels:
        level_dir = out_dir / level
        level_dir.mkdir(parents=True, exist_ok=True)

        for name in training_datasets:
            out = level_dir / f"documents_{name}.jsonl.gz"

            if out.exists() and not args.force:
                print(f"[{level}/{name}] already exists: {out}")
                continue

            src = Path(args.masked_dir) / level / DATASETS[name]["file"]

            if not src.exists():
                print(f"[{level}/{name}] file not found: {src}")
                continue

            labels = load_author_types(profiles, name)

            docs = build_documents(
                src, labels, name, args.max_comments, args.seed, cutoff
            )
            save_documents(docs, out)

            print(f"[{level}/{name}] {len(docs):,} documents -> {out}")

    if "final" in args.datasets:
        final_dir = out_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)

        out = final_dir / "documents_final.jsonl.gz"

        if out.exists() and not args.force:
            print(f"[final] already exists: {out}")
        else:
            src = Path(args.final_dir) / DATASETS["final"]["file"]

            if not src.exists():
                print(f"[final] file not found: {src}")
            else:
                labels = load_author_types(profiles, "final")

                docs = build_documents(
                    src, labels, "final", args.max_comments, args.seed, cutoff
                )
                save_documents(docs, out)

                print(f"[final] {len(docs):,} documents -> {out}")


if __name__ == "__main__":
    main()
