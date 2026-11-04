import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.masking import (MockBackend, OpenAICompatBackend, apply_spans, is_candidate,
                         parse_model_output, process_dataset_llm, validate_spans)


def test_parse_output():
    raw = '```json\n{"spans": [{"text": "INTJ", "category": "type"}]}\n```'
    assert parse_model_output(raw) == [{"text": "INTJ", "category": "type"}]
    assert parse_model_output("boh") is None
    assert parse_model_output('{"spans": "x"}') is None
    assert parse_model_output(None) is None


def test_validate_rejects_bad_spans():
    text = "I am an INTJ and I scored 85 on the test, honestly I like it a lot"
    spans = [
        {"text": "INTJ", "category": "type"},
        {"text": "intj", "category": "type"},            # duplicato (case)
        {"text": "unicorn", "category": "type"},         # non presente
        {"text": "I", "category": "type"},               # troppo corto / banale
        {"text": text, "category": "score"},             # tutto il commento
        {"text": "MENTION", "category": "type"},         # dentro un segnaposto
        {"text": "85", "category": "score"},
        {"text": 5, "category": "type"},                 # non stringa
    ]
    acc, rej = validate_spans(text, spans)
    assert [a[0] for a in acc] == ["INTJ", "85"]
    assert rej == 5


def test_apply_spans_boundaries():
    t = "scored 85 and 850, INTJ and INTJs"
    out = apply_spans(t, [("85", "score"), ("INTJ", "type")])
    assert out == "scored SCORE_MENTION and 850, TYPE_MENTION and INTJs"


def test_prefilter():
    assert is_candidate("I got INTJ on the test")
    assert is_candidate("my big five scores are in")
    assert is_candidate("What is your personality like?")
    assert not is_candidate("The weather today was great, we went hiking for hours")


def _mock_fn(text):
    spans = []
    if "INTJ" in text:
        spans.append({"text": "INTJ", "category": "type"})
    if "85" in text:
        spans.append({"text": "85", "category": "score"})
    if "HACK" in text:   # il modello "sbaglia": copre tutto il commento
        spans.append({"text": text, "category": "type"})
    return json.dumps({"spans": spans})


def test_pipeline_with_mock_and_cache(tmp_path):
    data, out = tmp_path / "d", tmp_path / "o"
    data.mkdir()
    rows = [
        ("a", "I am an INTJ and my friends are all very kind to me", "x"),
        ("b", "Weather was great and we went hiking for several hours", "y"),
        ("c", "My openness score was 85 on that personality test yesterday", "z"),
        ("d", "[deleted]", "x"),
        ("e", "HACK personality test whole comment is returned as a span", "x"),
    ]
    pd.DataFrame(rows, columns=["author", "body", "subreddit"]).to_csv(data / "comments_final.csv", index=False)

    be = MockBackend(_mock_fn)
    st = process_dataset_llm("final", data, out, be, "mock-model", chunk_size=2, batch_size=2)
    assert st["rows_kept"] == 4 and st["candidates"] == 3 and be.calls == 3
    res = pd.read_csv(out / "llm" / "comments_final.csv", keep_default_na=False)
    assert res.body[0].startswith("I am an TYPE_MENTION and")
    assert res.body[1] == rows[1][1]                          # non candidato: intatto
    assert "SCORE_MENTION" in res.body[2] and "openness" in res.body[2]
    assert res.body[3] == rows[4][1]                          # span "tutto il commento" scartato
    assert st["spans_rejected"] >= 1

    # seconda esecuzione: nessuna nuova chiamata, tutto dalla cache
    be2 = MockBackend(lambda t: (_ for _ in ()).throw(RuntimeError("non deve essere chiamato")))
    st2 = process_dataset_llm("final", data, out, be2, "mock-model", chunk_size=2, force=True)
    assert be2.calls == 0 and st2["cache_hits"] == 3 and st2["changed"] == st["changed"]

    # dry run: conta senza backend
    st3 = process_dataset_llm("final", data, out, None, "mock-model", dry_run=True)
    assert st3["unique_candidates"] == 3

    # pilota con limite: scrive in llm_pilot e non tocca llm
    st4 = process_dataset_llm("final", data, out, MockBackend(_mock_fn), "other-model", limit=1)
    assert (out / "llm_pilot" / "comments_final.csv").exists() and st4["llm_calls"] == 1


def test_openai_backend_with_fake_server():
    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert body["temperature"] == 0 and body["messages"][0]["role"] == "system"
            reply = {"choices": [{"message": {"content": '{"spans": []}'}}]}
            data = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        be = OpenAICompatBackend(f"http://127.0.0.1:{srv.server_port}/v1", "m", workers=2, retries=0)
        outs = be.generate([[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]] * 3)
        assert outs == ['{"spans": []}'] * 3
        dead = OpenAICompatBackend("http://127.0.0.1:9/v1", "m", retries=0, timeout=2)
        assert dead.generate([[{"role": "user", "content": "u"}]]) == [None]
    finally:
        srv.shutdown()


if __name__ == "__main__":
    import tempfile
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            if "tmp_path" in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
                with tempfile.TemporaryDirectory() as d:
                    fn(Path(d))
            else:
                fn()
            print("OK", name)
