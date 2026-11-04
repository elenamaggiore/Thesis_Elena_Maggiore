"""Backend per far girare l'LLM.

Tutti espongono  generate(batch_messages) -> list[str | None]  (None = chiamata fallita).

- OpenAICompatBackend: qualsiasi server con API /v1/chat/completions
  (Ollama, LM Studio, vLLM, ...). Consigliato per il PC: i commenti restano in locale.
- HFBackend: modello Hugging Face caricato in locale (per Colab con GPU).
- MockBackend: per i test, senza modello.
"""
from __future__ import annotations

import json
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


class OpenAICompatBackend:
    def __init__(self, base_url: str, model: str, api_key: str | None = None,
                 workers: int = 4, timeout: int = 180, max_tokens: int = 400, retries: int = 2):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.workers = workers
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.retries = retries

    def _one(self, messages: list[dict]) -> str | None:
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": self.max_tokens,
        }).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        for attempt in range(self.retries + 1):
            try:
                req = urllib.request.Request(self.url, data=payload, headers=headers)
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
            except Exception:
                if attempt < self.retries:
                    time.sleep(2 * (attempt + 1))
        return None

    def generate(self, batch_messages: list[list[dict]]) -> list[str | None]:
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            return list(ex.map(self._one, batch_messages))


class HFBackend:
    """Modello Hugging Face in locale (richiede torch + transformers, meglio con GPU)."""

    def __init__(self, model_name: str, batch_size: int = 16,
                 max_new_tokens: int = 300, load_in_4bit: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        kwargs = {"device_map": "auto", "torch_dtype": "auto"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.model.eval()

    def generate(self, batch_messages: list[list[dict]]) -> list[str | None]:
        prompts = [
            self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in batch_messages
        ]
        outs: list[str | None] = []
        for i in range(0, len(prompts), self.batch_size):
            enc = self.tok(prompts[i:i + self.batch_size], return_tensors="pt",
                           padding=True).to(self.model.device)
            with self._torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=self.max_new_tokens,
                                          do_sample=False)
            new = gen[:, enc["input_ids"].shape[1]:]
            outs.extend(self.tok.batch_decode(new, skip_special_tokens=True))
        return outs


class MockBackend:
    """fn(testo_del_commento) -> stringa di output (di solito un JSON)."""

    def __init__(self, fn):
        self.fn = fn
        self.calls = 0

    def generate(self, batch_messages: list[list[dict]]) -> list[str | None]:
        outs = []
        for msgs in batch_messages:
            self.calls += 1
            m = re.search(r"<comment>\n(.*)\n</comment>", msgs[-1]["content"], re.S)
            outs.append(self.fn(m.group(1) if m else ""))
        return outs
