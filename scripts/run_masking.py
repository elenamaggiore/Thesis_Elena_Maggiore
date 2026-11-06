"""Applica il masking a tutti i dataset PANDORA con un solo comando.

Output:  <out-dir>/<livello>/comments_<dataset>.csv   (+ masking_stats_<dataset>.json)
I file originali non vengono modificati.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.masking import DATASETS, LEVELS, load_author_types, process_dataset  # noqa: E402
from src.masking.rules import DROP_SUBREDDIT_LEVELS, load_excluded_subreddits  # noqa: E402

DEFAULT_EXCLUDED = Path(__file__).resolve().parents[1] / "configs" / "excluded_subreddits.txt"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR"), help="cartella dei comments_*.csv")
    ap.add_argument("--out-dir", default=os.environ.get("OUT_DIR"), help="cartella di output")
    ap.add_argument("--profiles", default="author_profiles.csv",
                    help="CSV con author, mbti e i cinque punteggi Big Five (relativo a --data-dir o assoluto)")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--levels", nargs="+", default=["full", "limited"], choices=list(LEVELS))
    ap.add_argument("--chunk-size", type=int, default=200_000)
    ap.add_argument("--min-words", type=int, default=5)
    ap.add_argument("--max-words", type=int, default=500)
    ap.add_argument("--excluded-subreddits", default=str(DEFAULT_EXCLUDED),
                    help="lista dei subreddit i cui commenti il livello FULL scarta del tutto (.txt o .csv)")
    ap.add_argument("--no-subreddit-filter", action="store_true",
                    help="disattiva lo scarto dei subreddit (il livello full tiene tutte le righe)")
    ap.add_argument("--force", action="store_true", help="rifa anche gli output gia' presenti")
    args = ap.parse_args()

    if not args.data_dir or not args.out_dir:
        ap.error("servono --data-dir e --out-dir (oppure le variabili DATA_DIR e OUT_DIR)")
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    profiles = Path(args.profiles)
    if not profiles.is_absolute():
        profiles = data_dir / profiles

    excluded = set()
    if not args.no_subreddit_filter and any(lv in DROP_SUBREDDIT_LEVELS for lv in args.levels):
        path = Path(args.excluded_subreddits)
        if not path.exists():
            ap.error(f"lista subreddit non trovata: {path} (usa --excluded-subreddits o --no-subreddit-filter)")
        excluded = load_excluded_subreddits(path)
        print(f"Subreddit esclusi dal livello full: {len(excluded)} ({path.name})")

    for name in args.datasets:
        rules = DATASETS[name]["rules"]
        # Big Five: il controllo NOT NULL sui cinque punteggi si fa sempre.
        # MBTI: il tipo dell'autore serve solo al livello limited.
        needs_labels = "bigfive" in rules or ("limited" in args.levels and "mbti" in rules)
        if needs_labels and not profiles.exists():
            ap.error(f"file dei profili non trovato: {profiles} (serve per il dataset {name!r})")
        author_types = load_author_types(profiles, name) if needs_labels else {}
        process_dataset(
            name, data_dir, out_dir, args.levels, author_types,
            chunk_size=args.chunk_size, min_words=args.min_words,
            max_words=args.max_words, force=args.force, excluded_subreddits=excluded,
        )


if __name__ == "__main__":
    main()
