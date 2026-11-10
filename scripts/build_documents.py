"""Fase 1: un documento per autore (con i target binari) dai commenti MASCHERATI.
Per default elabora i QUATTRO livelli di masking (none, limited, full, llm): ognuno e' un
dataset di partenza diverso per l'oversampling.
Output: <out-dir>/documents_<livello>_<dataset>.jsonl.gz
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.documents import build_documents, save_documents  # noqa: E402
from src.masking.pipeline import DATASETS, load_author_types  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--masked-dir", required=True, help="cartella con le sottocartelle full/, limited/, ...")
    ap.add_argument("--profiles", required=True, help="author_profiles.csv")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--levels", nargs="+", default=["none", "limited", "full", "llm"],
                    help="cartelle di masking da usare (default: none limited full llm)")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--max-comments", type=int, default=200, help="tetto di commenti per autore")
    ap.add_argument("--bigfive-cutoff", type=float, default=50.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    for level in args.levels:
        for name in args.datasets:
            out = Path(args.out_dir) / f"documents_{level}_{name}.jsonl.gz"
            if out.exists() and not args.force:
                print(f"[{level}/{name}] gia' presente: {out} (usa --force per rifarlo)")
                continue
            src = Path(args.masked_dir) / level / DATASETS[name]["file"]
            if not src.exists():
                print(f"[{level}/{name}] file non trovato: {src}")
                continue
            labels = load_author_types(Path(args.profiles), name)
            docs = build_documents(src, labels, name, args.max_comments, args.seed, args.bigfive_cutoff)
            save_documents(docs, out)
            print(f"[{level}/{name}] {len(docs):,} documenti (su {len(labels):,} autori con etichette) -> {out}")


if __name__ == "__main__":
    main()
