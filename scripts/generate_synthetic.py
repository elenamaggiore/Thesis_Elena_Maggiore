
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.documents import load_documents
from src.data.targets import TARGETS
from src.sampling.gpt2_aug import (
    GPT2Generator,
    SyntheticCache,
    augment_minority,
    make_input_fn,
    plan_synthetic,
)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    ap.add_argument("--docs-dir", required=True)
    ap.add_argument("--synthetic-dir", default=None)

    ap.add_argument(
        "--levels",
        nargs="+",
        default=["none", "limited", "full", "llm"]
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Count documents and chunks without loading GPT-2"
    )
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=["mbti", "big_five"],
        choices=["mbti", "big_five"]
    )
    ap.add_argument("--targets", nargs="+", default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--multiplier", type=float, default=1.0)
    ap.add_argument("--chunk-size", type=int, default=5)
    ap.add_argument("--max-chunks", type=int, default=8)

    ap.add_argument(
        "--input",
        nargs="+",
        default=["full"],
        choices=["full", "minimal", "raw"]
    )

    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    docs_dir = Path(args.docs_dir)

    synthetic_dir = (
        Path(args.synthetic_dir)
        if args.synthetic_dir
        else docs_dir.parent / "Datasets_synthetic"
    )

    if args.dry_run:
        dry_run(args, docs_dir)
        return

    if not args.cache:
        ap.error("Specify --cache unless using --dry-run")

    synthetic_dir.mkdir(parents=True, exist_ok=True)

    gen = GPT2Generator(
        args.model,
        batch_size=args.batch_size,
        seed=args.seed
    )

    cache = SyntheticCache(args.cache)

    try:
        for level in args.levels:
            for name in args.datasets:

                path = (
                    docs_dir
                    / level
                    / f"documents_{name}.jsonl.gz"
                )

                if not path.exists():
                    print(f"[{level}/{name}] Documents not found: {path}")
                    continue

                docs = load_documents(path)

                for target in (args.targets or TARGETS[name]):

                    if target not in docs.columns:
                        print(
                            f"[{level}/{name}/{target}] "
                            "Target not found"
                        )
                        continue

                    for kind in args.input:

                        input_fn, tag = make_input_fn(kind)

                        syn = augment_minority(
                            docs,
                            target,
                            gen,
                            multiplier=args.multiplier,
                            chunk_size=args.chunk_size,
                            max_chunks=args.max_chunks or None,
                            cache=cache,
                            seed=args.seed,
                            input_fn=input_fn,
                            input_tag=tag
                        )

                        syn = syn.copy()

                        syn["synthetic_target"] = target
                        syn["input_strategy"] = kind
                        syn["masking_level"] = level
                        syn["dataset"] = name

                        out_dir = (
                            synthetic_dir
                            / level
                            / name
                        )

                        out_dir.mkdir(
                            parents=True,
                            exist_ok=True
                        )

                        out_path = (
                            out_dir
                            / f"{target}_{kind}.csv"
                        )

                        if "comments" in syn.columns:
                            syn["comments"] = syn["comments"].apply(
                                lambda x: json.dumps(
                                    x,
                                    ensure_ascii=False
                                ) if isinstance(x, list) else x
                            )

                        syn.to_csv(
                            out_path,
                            index=False,
                            encoding="utf-8-sig"
                        )

                        print(
                            f"[{level}/{name}/{target}/input={kind}] "
                            f"{len(syn):,} synthetic documents -> "
                            f"{out_path}",
                            flush=True
                        )

    finally:
        cache.close()


def dry_run(args, docs_dir):
    total_docs = 0
    total_chunks = 0

    cap = args.max_chunks or 10 ** 9

    for level in args.levels:
        for name in args.datasets:

            path = (
                docs_dir
                / level
                / f"documents_{name}.jsonl.gz"
            )

            if not path.exists():
                print(f"[{level}/{name}] Documents not found: {path}")
                continue

            docs = load_documents(path)
            picks = set()

            for target in (args.targets or TARGETS[name]):

                if target in docs.columns:
                    for pos, copy_idx in plan_synthetic(
                        docs[target].to_numpy(),
                        args.multiplier,
                        args.seed
                    ):
                        picks.add((pos, copy_idx))

            chunks = sum(
                min(
                    -(-len(docs.iloc[pos]["comments"]) // args.chunk_size),
                    cap
                )
                for pos, _ in picks
            )

            print(
                f"[{level}/{name}] "
                f"{len(picks):,} synthetic documents, "
                f"{chunks:,} chunks per input"
            )

            total_docs += len(picks)
            total_chunks += chunks

    k = len(args.input)

    print(
        f"TOTAL: {total_docs * k:,} documents, "
        f"{total_chunks * k:,} GPT-2 generations "
        f"({len(args.levels)} levels x {k} inputs: "
        f"{', '.join(args.input)})"
    )


if __name__ == "__main__":
    main()
