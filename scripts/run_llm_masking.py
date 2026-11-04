"""Masking con LLM (l'LLM indica gli span, il nostro codice li applica).

Passi consigliati:
  1) conta quanti commenti andrebbero all'LLM, senza chiamarlo:
       python scripts/run_llm_masking.py --data-dir ... --out-dir ... --datasets final --dry-run
  2) pilota su pochi commenti, poi controlla llm_audit_<dataset>_pilot.csv:
       python scripts/run_llm_masking.py ... --datasets final --backend openai \
           --base-url http://localhost:11434/v1 --model qwen2.5:7b-instruct --limit 300
  3) run completo (riprende da dove si e' fermato grazie alla cache):
       python scripts/run_llm_masking.py ... --datasets final --backend openai ...

Output: <out-dir>/llm/comments_<dataset>.csv  (oppure llm_hybrid/ con --hybrid)
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.masking import DATASETS, HFBackend, OpenAICompatBackend, process_dataset_llm  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR"))
    ap.add_argument("--out-dir", default=os.environ.get("OUT_DIR"))
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--backend", choices=["openai", "hf"], default="openai",
                    help="openai = server locale compatibile (Ollama, LM Studio, vLLM); hf = Hugging Face in locale")
    ap.add_argument("--model", help="nome del modello (es. qwen2.5:7b-instruct su Ollama, o Qwen/Qwen2.5-7B-Instruct su hf)")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--api-key-env", default=None, help="nome della variabile d'ambiente con la chiave API (di solito non serve in locale)")
    ap.add_argument("--workers", type=int, default=4, help="richieste parallele (backend openai)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--hf-4bit", action="store_true", help="quantizzazione a 4 bit (backend hf, richiede bitsandbytes)")
    ap.add_argument("--hybrid", action="store_true", help="dopo l'LLM applica anche le regex FULL (output in llm_hybrid/)")
    ap.add_argument("--dry-run", action="store_true", help="conta i commenti candidati senza chiamare l'LLM")
    ap.add_argument("--limit", type=int, default=None, help="pilota: manda all'LLM solo N commenti (output in llm_pilot/)")
    ap.add_argument("--audit-n", type=int, default=200)
    ap.add_argument("--chunk-size", type=int, default=50_000)
    ap.add_argument("--min-words", type=int, default=5)
    ap.add_argument("--max-words", type=int, default=500)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not args.data_dir or not args.out_dir:
        ap.error("servono --data-dir e --out-dir (oppure DATA_DIR e OUT_DIR)")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    backend = None
    if not args.dry_run:
        if not args.model:
            ap.error("serve --model (tranne con --dry-run)")
        if args.backend == "openai":
            key = os.environ.get(args.api_key_env) if args.api_key_env else None
            backend = OpenAICompatBackend(args.base_url, args.model, key, workers=args.workers)
        else:
            backend = HFBackend(args.model, batch_size=args.batch_size, load_in_4bit=args.hf_4bit)

    for name in args.datasets:
        process_dataset_llm(
            name, Path(args.data_dir), Path(args.out_dir), backend, args.model or "none",
            hybrid=args.hybrid, chunk_size=args.chunk_size, min_words=args.min_words,
            max_words=args.max_words, batch_size=args.batch_size, force=args.force,
            dry_run=args.dry_run, limit=args.limit, audit_n=args.audit_n,
        )


if __name__ == "__main__":
    main()
