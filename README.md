# Thesis_Elena_Maggiore
This is the Git folder for my Master Thesis, in which I will perform a coomparison between the predictive power of Big Five and MBTI over a corpus of comments by Reddit users.

To execute code:

* Download the project folder:
* Open Python
* python tests/test_masking.py
* python scripts/run_masking.py \
    --data-dir ~/Desktop/dati \
    --out-dir ~/Desktop/risultati \
    --datasets big_five
* python scripts/run_masking.py \
    --data-dir ~/Desktop/dati \
    --out-dir ~/Desktop/risultati \
    --datasets mbti
* curl -fsSL https://ollama.com/install.sh | sh
* ollama --version
* ollama pull qwen2.5:7b-instruct
* sudo systemctl start ollama
* python scripts/run_llm_masking.py \
    --data-dir ~/Desktop/dati \
    --out-dir ~/Desktop/risultati \
    --datasets big_five \
    --dry-run
* python scripts/run_llm_masking.py \
    --data-dir ~/Desktop/dati \
    --out-dir ~/Desktop/risultati \
    --datasets mbti \
    --dry-run
* python scripts/run_llm_masking.py \
    --data-dir ~/Desktop/dati \
    --out-dir ~/Desktop/risultati \
    --datasets big_five \
    --backend openai \
    --base-url http://localhost:11434/v1 \
    --model qwen2.5:7b-instruct
* python scripts/run_llm_masking.py \
    --data-dir ~/Desktop/dati \
    --out-dir ~/Desktop/risultati \
    --datasets mbti \
    --backend openai \
    --base-url http://localhost:11434/v1 \
    --model qwen2.5:7b-instruct
