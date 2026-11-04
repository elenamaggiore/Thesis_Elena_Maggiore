"""Applica il masking a tutti i dataset PANDORA con un solo comando.

Esempio (Colab):
  python scripts/run_masking.py \
      --data-dir "/content/drive/MyDrive/Thesis_Elena_Maggiore/Datasets" \
      --out-dir  "/content/drive/MyDrive/Thesis_Elena_Maggiore/Datasets_masked"

Output:  <out-dir>/<livello>/comments_<dataset>.csv   (+ masking_stats_<dataset>.json)
I file originali non vengono modificati.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.masking import DATASETS, LEVELS, load_author_types, process_dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR"), help="cartella dei comments_*.csv")
    ap.add_argument("--out-dir", default=os.environ.get("OUT_DIR"), help="cartella di output")
    ap.add_argument("--profiles", default="author_profiles.csv",
                    help="CSV con author e mbti (relativo a --data-dir o percorso assoluto)")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--levels", nargs="+", default=["full", "limited"], choices=list(LEVELS))
    ap.add_argument("--chunk-size", type=int, default=200_000)
    ap.add_argument("--min-words", type=int, default=5)
    ap.add_argument("--max-words", type=int, default=500)
    ap.add_argument("--force", action="store_true", help="rifa anche gli output gia' presenti")
    args = ap.parse_args()

    if not args.data_dir or not args.out_dir:
        ap.error("servono --data-dir e --out-dir (oppure le variabili DATA_DIR e OUT_DIR)")
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    profiles = Path(args.profiles)
    if not profiles.is_absolute():
        profiles = data_dir / profiles
    needs_types = "limited" in args.levels and any("mbti" in DATASETS[d]["rules"] for d in args.datasets)
    author_types = load_author_types(profiles) if needs_types else {}
    if needs_types:
        print(f"Tipi MBTI noti per {len(author_types):,} autori ({profiles.name})")

    for name in args.datasets:
        process_dataset(
            name, data_dir, out_dir, args.levels, author_types,
            chunk_size=args.chunk_size, min_words=args.min_words,
            max_words=args.max_words, force=args.force,
        )


if __name__ == "__main__":
    main()
