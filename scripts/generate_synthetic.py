"""Genera con GPT-2 i documenti sintetici e li salva in cache.

Il testo cambia da un livello di masking all'altro, quindi la generazione si fa PER LIVELLO
(i documenti identici fra livelli riusano la stessa cache). Con --dry-run conta quanti
documenti e blocchi andrebbero generati, senza caricare GPT-2: fallo PRIMA, il costo cresce
con livelli x input.

Da lanciare su GPU (Colab). Per ogni target genera un documento sintetico per ogni utente
della classe minoritaria (moltiplicatore 1.0). Le fasi successive (prepare_target) trovano
tutto in cache. Se si interrompe, rilanciare lo stesso comando: riparte dalla cache.

Esempio:
  python scripts/generate_synthetic.py --docs-dir "C:\\...\\Documents" --dry-run
  python scripts/generate_synthetic.py --docs-dir "C:\\...\\Documents" \
      --cache "C:\\...\\Documents\\gpt2_cache.sqlite"
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.documents import load_documents  # noqa: E402
from src.data.targets import TARGETS  # noqa: E402
from src.sampling.gpt2_aug import (GPT2Generator, SyntheticCache, augment_minority,  # noqa: E402
                                   make_input_fn, plan_synthetic)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs-dir", required=True)
    ap.add_argument("--levels", nargs="+", default=["none", "limited", "full", "llm"])
    ap.add_argument("--dry-run", action="store_true", help="conta i documenti/blocchi da generare, senza GPT-2")
    ap.add_argument("--datasets", nargs="+", default=["mbti", "big_five"], choices=["mbti", "big_five"])
    ap.add_argument("--targets", nargs="+", default=None, help="solo alcuni target (default: tutti)")
    ap.add_argument("--cache", help="file SQLite della cache (obbligatorio senza --dry-run)")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--multiplier", type=float, default=1.0)
    ap.add_argument("--chunk-size", type=int, default=5, help="commenti per blocco (paper: 5)")
    ap.add_argument("--max-chunks", type=int, default=8, help="blocchi per documento (0 = tutti)")
    ap.add_argument("--input", nargs="+", default=["full"], choices=["full", "minimal", "raw"],
                    help="preprocessing da applicare ai commenti PRIMA di darli a GPT-2 (default: full, "
                         "come nel paper). 'minimal' serve alla strategia noleak_minimal. "
                         "Ogni valore e' una generazione separata (costo pieno).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.dry_run:
        dry_run(args)
        return
    if not args.cache:
        ap.error("serve --cache (tranne con --dry-run)")
    gen = GPT2Generator(args.model, batch_size=args.batch_size, seed=args.seed)
    cache = SyntheticCache(args.cache)
    for level in args.levels:
        for name in args.datasets:
            path = Path(args.docs_dir) / f"documents_{level}_{name}.jsonl.gz"
            if not path.exists():
                print(f"[{level}/{name}] documenti non trovati: {path}")
                continue
            docs = load_documents(path)
            for target in (args.targets or TARGETS[name]):
                if target not in docs.columns:
                    continue
                for kind in args.input:
                    input_fn, tag = make_input_fn(kind)
                    syn = augment_minority(docs, target, gen, multiplier=args.multiplier,
                                           chunk_size=args.chunk_size, max_chunks=args.max_chunks or None,
                                           cache=cache, seed=args.seed, input_fn=input_fn, input_tag=tag)
                    print(f"[{level}/{name}/{target}/input={kind}] sintetici disponibili: {len(syn):,}", flush=True)
    cache.close()


def dry_run(args):
    """Conta il numero di generazioni necessarie (un documento sintetico = un blocco per ogni 5 commenti, max --max-chunks)."""
    total_docs = total_chunks = 0
    cap = args.max_chunks or 10 ** 9
    for level in args.levels:
        for name in args.datasets:
            path = Path(args.docs_dir) / f"documents_{level}_{name}.jsonl.gz"
            if not path.exists():
                print(f"[{level}/{name}] documenti non trovati: {path}")
                continue
            docs = load_documents(path)
            picks = set()                    # la cache e' per (autore, copia): un autore conta una volta
            for target in (args.targets or TARGETS[name]):
                if target in docs.columns:
                    for pos, c in plan_synthetic(docs[target].to_numpy(), args.multiplier, args.seed):
                        picks.add((pos, c))
            chunks = sum(min(-(-len(docs.iloc[pos]["comments"]) // args.chunk_size), cap) for pos, _ in picks)
            print(f"[{level}/{name}] {len(picks):,} documenti sintetici, {chunks:,} blocchi per input")
            total_docs += len(picks)
            total_chunks += chunks
    k = len(args.input)
    print(f"TOTALE: {total_docs * k:,} documenti, {total_chunks * k:,} generazioni GPT-2 "
          f"({len(args.levels)} livelli x {k} input: {', '.join(args.input)})")


if __name__ == "__main__":
    main()
