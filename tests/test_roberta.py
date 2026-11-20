"""Test di RoBERTa multi-task, 20 dataset, metriche e valutazione.

I test con il modello usano un RoBERTa MINUSCOLO creato sul momento (senza scaricare nulla);
se torch/transformers non sono installati vengono saltati (SKIP) e passano gli altri.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn import metrics as sk

from src.data.targets import BIG_FIVE_TARGETS, MBTI_TARGETS
from src.eval.metrics import (bootstrap_indices, bootstrap_metrics, binary_metrics, ci95, paired_difference,
                              mean_over_tasks, task_metrics)
from src.models.chunks import ChunkPool, build_chunks, build_pool, evenly_spaced
from src.sampling.gpt2_aug import MockGenerator, SyntheticCache, make_input_fn
from src.sampling.multitask import MASKING_LEVELS, all_datasets, build_multitask_dataset, multilabel_split
from src.sampling.undersample import undersample_strata

try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False

WORDS = "cat dog run jump think feel plan idea book game music night day friend work home love time".split()
ALL_TARGETS = list(MBTI_TARGETS) + list(BIG_FIVE_TARGETS)


def make_docs(n=100, seed=0, n_comments=12):
    rnd = np.random.default_rng(seed)
    rows = []
    rates = [0.2, 0.3, 0.5, 0.7, 0.25, 0.4, 0.5, 0.6, 0.35]            # squilibri diversi per target
    for i in range(n):
        y = [int(rnd.random() < r) for r in rates]
        comments = []
        for _ in range(n_comments):
            sig = [f"sig{j}{'p' if y[j] else 'n'}" for j in range(len(y)) if rnd.random() < 0.3]
            comments.append(" ".join(list(rnd.choice(WORDS, 6)) + sig))
        row = {"author": f"u{i}", "comments": comments, "n_comments_seen": n_comments,
               "text": " ||| ".join(comments), "source_author": f"u{i}", "is_synthetic": False}
        row.update(dict(zip(ALL_TARGETS, y)))
        rows.append(row)
    return pd.DataFrame(rows)


def mock_gen():
    g = MockGenerator(lambda t: "generated " + " ".join(reversed(t.split()[:8])))
    g.signature = "mock"
    return g


# ----------------------------------------------------------------------------- metriche (senza torch)
def test_metrics_match_sklearn_and_bootstrap():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    p = np.clip(y * 0.3 + rng.random(300) * 0.7, 0, 1)
    m, pred = binary_metrics(y, p), (p >= 0.5).astype(int)
    ref = dict(accuracy=sk.accuracy_score(y, pred), balanced_accuracy=sk.balanced_accuracy_score(y, pred),
               macro_f1=sk.f1_score(y, pred, average="macro"), f1_pos=sk.f1_score(y, pred),
               mcc=sk.matthews_corrcoef(y, pred), auc=sk.roc_auc_score(y, p))
    assert all(abs(m[k] - ref[k]) < 1e-12 for k in ref)
    # una sola classe presente: AUC non definita, niente errori
    assert np.isnan(binary_metrics([1, 1, 1], [0.2, 0.9, 0.7])["auc"])
    # bootstrap appaiato: stesso ricampionamento per due famiglie -> differenza nulla se uguali
    Y = rng.integers(0, 2, (120, 3))
    P = np.clip(Y * 0.3 + rng.random((120, 3)) * 0.7, 0, 1)
    idx = bootstrap_indices(120, 200, 1)
    assert idx.shape == (200, 120) and (idx == bootstrap_indices(120, 200, 1)).all()
    _, ma = bootstrap_metrics(Y, P, ["a", "b", "c"], idx)
    _, mb = bootstrap_metrics(Y, P, ["x", "y", "z"], idx)
    d = paired_difference(ma, mb)
    assert d["auc"]["diff"] == 0 and d["auc"]["ci_low"] == 0 and not d["auc"]["significant"]
    lo, hi = ci95(ma["auc"])
    assert lo < mean_over_tasks(task_metrics(Y, P, ["a", "b", "c"]))["auc"] < hi


# ----------------------------------------------------------------------------- chunk (senza torch)
def test_chunks_and_epoch_resampling():
    count = lambda cs: [len(c.split()) for c in cs]
    ch = build_chunks(["a b c", "d e", "f g h i", "", "j"], count([ "a b c", "d e", "f g h i", "", "j"]), max_tokens=6)
    assert ch == ["a b c d e", "f g h i j"]
    assert build_chunks(["x " * 10], [10], max_tokens=6) == ["x " * 10]       # commento piu' lungo: chunk da solo
    assert evenly_spaced(10, 3) == [0, 4, 9] and evenly_spaced(2, 5) == [0, 1]
    pool = ChunkPool(["big", "small"], [[f"c{i}" for i in range(100)], ["s0", "s1", "s2"]],
                     np.array([[1, 0], [0, 1]], dtype=np.float32))
    items = pool.sample_epoch(np.random.default_rng(0), 4)
    big = [t for t, _ in items if t.startswith("c")]
    assert len(big) == 4 and len(items) == 4 + 3 == pool.epoch_size(4)      # nessun autore domina: al piu' k chunk
    assert len(set(big)) == 4                                                # senza reinserimento
    other = pool.sample_epoch(np.random.default_rng(1), 4)
    assert [t for t, _ in other if t.startswith("c")] != big                 # ogni epoca ricampiona
    assert [t for t, _ in pool.sample_epoch(np.random.default_rng(0), 4)] == [t for t, _ in items]   # riproducibile
    # build_pool: etichette dell'autore su ogni chunk e tetto del pool
    frame = pd.DataFrame([{"author": "a", "comments": ["w " * 5] * 30, "introverted": 1, "intuitive": 0}])
    p = build_pool(frame, ["introverted", "intuitive"], lambda cs: [5] * len(cs), max_tokens=10, cap=4)
    assert len(p.chunks[0]) == 4 and p.labels.tolist() == [[1.0, 0.0]]


# ----------------------------------------------------------------------------- 20 dataset (senza torch)
def test_twenty_datasets_and_splits():
    assert len(all_datasets()) == 20 and len(MASKING_LEVELS) == 4
    assert ("full", "noleak_full") in all_datasets()
    Y = np.array([[1, 0]] * 40 + [[0, 1]] * 40 + [[1, 1]])                    # uno strato con 1 solo autore
    tr, va = multilabel_split(Y, 0.2, 1)
    assert len(set(tr) & set(va)) == 0 and len(tr) + len(va) == 81
    Yu = np.array([[1, 1]] * 100 + [[0, 0]] * 6 + [[1, 0]] * 8 + [[0, 1]] * 10)
    keep = undersample_strata(Yu, "median", 0)
    keys = ["".join(map(str, r)) for r in Yu[keep]]
    assert keys.count("11") == 9 and keys.count("00") == 6                    # mediana degli strati = 9


def test_multitask_strategies():
    docs = make_docs(100)
    cache, gen = SyntheticCache(), mock_gen()
    kw = dict(generator=gen, cache=cache, max_chunks=2, split_seed=3)
    out = {s: build_multitask_dataset(docs, MBTI_TARGETS, s, **kw) for s in
           ("paper_leaky", "noleak_full", "noleak_minimal", "no_oversampling", "undersampling")}
    r = {s: d.report for s, d in out.items()}
    assert r["paper_leaky"]["val_rows_with_source_in_train"] > 0 and r["paper_leaky"]["val_synthetic_rows"] > 0
    for s in ("noleak_full", "noleak_minimal", "no_oversampling", "undersampling"):
        assert r[s]["val_rows_with_source_in_train"] == 0 and r[s]["val_synthetic_rows"] == 0
        assert not set(out[s].train.author) & set(out[s].val.author)
    assert r["no_oversampling"]["synthetic_in_train"] == 0 and r["no_oversampling"]["n_train"] == 80
    assert r["undersampling"]["n_train"] < r["no_oversampling"]["n_train"] and r["undersampling"]["synthetic_in_train"] == 0
    assert r["noleak_full"]["synthetic_in_train"] > 0 and r["noleak_full"]["gpt2_input"] == "full"
    assert r["noleak_minimal"]["gpt2_input"] == "minimal" and r["no_oversampling"]["gpt2_input"] is None
    # i sintetici ereditano TUTTE le etichette dell'autore di partenza e sono unici per autore
    syn = out["noleak_full"].train[out["noleak_full"].train.is_synthetic]
    assert syn.author.is_unique and syn.author.str.contains("#syn").all()
    base = docs.set_index("author")
    for _, row in syn.iterrows():
        assert [row[t] for t in MBTI_TARGETS] == [base.loc[row["source_author"], t] for t in MBTI_TARGETS]
    # Big Five: 5 target nello stesso modo
    bf = build_multitask_dataset(docs, BIG_FIVE_TARGETS, "noleak_full", **kw)
    assert bf.targets == list(BIG_FIVE_TARGETS) and bf.report["val_rows_with_source_in_train"] == 0
    # senza generatore e senza cache le strategie GPT-2 si fermano con un messaggio chiaro
    try:
        build_multitask_dataset(docs, MBTI_TARGETS, "noleak_full", generator=None, cache=SyntheticCache(), max_chunks=2)
        assert False
    except RuntimeError as e:
        assert "generate_synthetic" in str(e)


# ----------------------------------------------------------------------------- RoBERTa (con torch)
def make_tiny_encoder(path: Path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, RobertaConfig, RobertaModel
    vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}
    for w in WORDS + [f"sig{j}{c}" for j in range(9) for c in "pn"] + ["generated"]:
        vocab.setdefault(w, len(vocab))
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", pad_token="<pad>", cls_token="<s>",
                            sep_token="</s>", bos_token="<s>", eos_token="</s>").save_pretrained(str(path))
    RobertaModel(RobertaConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                               intermediate_size=64, max_position_embeddings=140, type_vocab_size=1,
                               pad_token_id=1, bos_token_id=0, eos_token_id=2)).save_pretrained(str(path))


TINY_CFG = dict(epochs=3, patience=2, batch_size=8, max_len=48, chunks_per_author=2, max_pool_chunks=4,
                max_eval_chunks=4, amp=False, lr_encoder=1e-3, lr_head=1e-2)


def test_train_evaluate_and_ensemble(tmp_path):
    if not HAVE_TORCH:
        print("SKIP (torch/transformers non installati)")
        return
    from src.eval.evaluate import evaluate_checkpoints, results_table
    from src.models.roberta import compute_pos_weight, train_run

    enc = tmp_path / "tiny"
    make_tiny_encoder(enc)
    docs, test = make_docs(60, seed=1), make_docs(30, seed=2)
    test["author"] = [f"t{i}" for i in range(len(test))]
    # pos_weight limitata tra 0.5 e 3
    Y = np.array([[1, 0, 1]] * 2 + [[0, 1, 0]] * 98)
    assert compute_pos_weight(Y, 0.5, 3.0).tolist() == [3.0, 0.5, 3.0]

    def fit(targets, seed, name, **extra):
        data = build_multitask_dataset(docs, targets, "no_oversampling")
        cfg = {**TINY_CFG, "encoder": str(enc), "seed": seed, "level": "full", "strategy": "no_oversampling",
               "dataset": "mbti" if targets == MBTI_TARGETS else "big_five", **extra}
        out = tmp_path / name
        s = train_run(cfg, data, out, log=lambda m: None)
        return out, s

    a1, s1 = fit(MBTI_TARGETS, 1, "mbti_s1")
    a2, _ = fit(MBTI_TARGETS, 2, "mbti_s2")
    b1, _ = fit(BIG_FIVE_TARGETS, 1, "bf_s1")
    for d in (a1, a2, b1):
        for f in ("best.pt", "config.json", "history.json", "done.json", "val_probs.npz"):
            assert (d / f).exists(), f
    cfg = json.loads((a1 / "config.json").read_text())
    assert cfg["n_tasks"] == 4 and cfg["dropout"] == 0.3 and cfg["targets"] == list(MBTI_TARGETS)
    hist = json.loads((a1 / "history.json").read_text())
    assert 1 <= len(hist) <= 3 and np.isfinite(hist[0]["train_loss"]) and "mean_auc" in hist[0]
    assert s1["best_epoch"] >= 1
    assert json.loads((b1 / "config.json").read_text())["n_tasks"] == 5                    # Big Five: 5 uscite
    # early stopping: con lr = 0 il modello non cambia, quindi si ferma dopo patience epoche senza miglioramento
    z, sz = fit(MBTI_TARGETS, 3, "zero_lr", lr_encoder=0.0, lr_head=0.0, epochs=6, patience=1)
    assert sz["epochs_run"] == 2 and sz["best_epoch"] == 1
    # valutazione: singolo, ensemble di due seed (model averaging), confronto appaiato Big Five vs MBTI
    single = evaluate_checkpoints([str(a1)], test, B=60)
    ens = evaluate_checkpoints([str(a1), str(a2)], test, B=60)
    assert single["a"]["n_authors"] == ens["a"]["n_authors"] == 30
    tab = results_table(ens["a"])
    assert set(tab.metric) == {"accuracy", "balanced_accuracy", "macro_f1", "f1_pos", "mcc", "auc"}
    assert set(tab.task) == set(MBTI_TARGETS) | {"MEDIA"} and (tab.ci_low <= tab.ci_high).all()
    pair = evaluate_checkpoints([str(b1)], test, pair_with=[str(a1)], B=60)
    d = pair["paired_difference"]["auc"]
    assert abs(d["diff"] - (pair["a"]["point_mean"]["auc"] - pair["b"]["point_mean"]["auc"])) < 1e-12
    assert d["ci_low"] <= d["ci_high"] and pair["a"]["n_authors"] == pair["b"]["n_authors"]
    # un ensemble richiede modelli identici: mescolare famiglie da' errore
    try:
        evaluate_checkpoints([str(a1), str(b1)], test, B=10)
        assert False
    except ValueError as e:
        assert "ensemble" in str(e)
    # le predizioni sono in cache nella cartella del checkpoint
    assert (a1 / "test_probs.npz").exists()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            if "tmp_path" in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
                with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
                    fn(Path(d))
            else:
                fn()
            print("OK", name)
