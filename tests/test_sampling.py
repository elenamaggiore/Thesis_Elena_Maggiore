import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.data.documents import build_documents, load_documents, save_documents
from src.data.split import stratified_split
from src.data.targets import add_targets
from src.features.preprocess import expand_contractions, get_preprocessor, preprocess_minimal
from src.masking.pipeline import load_author_types
from src.sampling.gpt2_aug import (CacheOnlyGenerator, MockGenerator, SyntheticCache, augment_minority,
                                   make_chunks, make_input_fn, make_signature, plan_synthetic)
from src.sampling.prepare import prepare_target
from src.sampling.smote import oversample_vectors
from src.sampling.strategies import STRATEGIES, Strategy
from src.sampling.undersample import undersample_indices

W2V = dict(vector_size=16, epochs=3, workers=1, min_count=1)
WORDS = "cat dog run jump think feel plan idea book game music night day friend work home love time".split()


def make_docs(n_pos=12, n_neg=48, seed=0, target="introverted"):
    rnd = random.Random(seed)
    rows = []
    for i in range(n_pos + n_neg):
        y = int(i < n_pos)
        comments = [" ".join(rnd.choices(WORDS, k=8)) for _ in range(10)]
        rows.append({"author": f"u{i}", "comments": comments, "n_comments_seen": 10,
                     "text": " ||| ".join(comments), target: y, "source_author": f"u{i}",
                     "is_synthetic": False})
    return pd.DataFrame(rows)


def mock_gen():
    return MockGenerator(lambda t: "generated " + " ".join(reversed(t.split()[:10])))


def test_targets_and_documents(tmp_path):
    profiles = tmp_path / "p.csv"
    pd.DataFrame({"author": ["a", "b"], "mbti": ["intj", "esfp"], "openness": ["80", "20"],
                  "conscientiousness": ["10", "90"], "extraversion": ["50", "51"],
                  "agreeableness": ["50", "49"], "neuroticism": ["0", "100"]}).to_csv(profiles, index=False)
    labels = load_author_types(profiles, "final")
    rows = [("a", f"comment number {i} from user a", "x") for i in range(30)] + \
           [("b", f"comment number {i} from user b", "x") for i in range(3)] + \
           [("zzz", "someone without labels at all", "x")]
    csv = tmp_path / "comments_final.csv"
    pd.DataFrame(rows, columns=["author", "body", "subreddit"]).to_csv(csv, index=False)
    docs = build_documents(csv, labels, "final", max_comments=10, seed=1)
    assert list(docs.author) == ["a", "b"]                       # "zzz" senza etichette escluso
    a = docs[docs.author == "a"].iloc[0]
    assert len(a.comments) == 10 and a.n_comments_seen == 30      # tetto rispettato
    assert (a.introverted, a.intuitive, a.thinking, a.perceiving) == (1, 1, 1, 0)   # INTJ
    b = docs[docs.author == "b"].iloc[0]
    assert (b.introverted, b.intuitive, b.thinking, b.perceiving) == (0, 0, 0, 1)   # ESFP
    assert (a.openness, b.openness) == (1, 0)                     # >50
    assert docs.set_index("author").loc["a", "extraversion"] == 0 and docs.set_index("author").loc["b", "extraversion"] == 1
    save_documents(docs, tmp_path / "d.jsonl.gz")
    back = load_documents(tmp_path / "d.jsonl.gz")
    assert isinstance(back.comments[0], list) and len(back) == 2
    # riproducibile
    again = build_documents(csv, labels, "final", max_comments=10, seed=1)
    assert list(again.comments[0]) == list(docs.comments[0])


def test_preprocessing():
    assert expand_contractions("I can't, don't and it's fine. I'm sure") == \
        "I cannot, do not and it is fine. I am sure"
    assert preprocess_minimal("Hello, World! It's 5 TYPE_MENTION") == ["hello", "world", "it's", "5", "type_mention"]
    full = get_preprocessor("full")
    toks = full("I don't like ENTPs. TYPE_MENTION is the best, running dogs! http://x.com @bob")
    assert "i" in toks and "like" in toks and "dog" in toks         # lemma: dogs -> dog
    assert "typemention" in toks                                    # segnaposto protetto
    assert "entps" not in toks and "bob" not in toks and "the" not in toks and "x" not in toks


def test_split_and_undersample_and_smote():
    y = [1] * 10 + [0] * 40
    tr, va = stratified_split(y, 0.2, 1)
    assert len(set(tr) & set(va)) == 0 and np.asarray(y)[va].sum() == 2
    keep = undersample_indices(y, seed=1)
    assert len(keep) == 20 and np.asarray(y)[keep].sum() == 10
    X = np.random.default_rng(0).normal(size=(50, 4))
    Xr, yr, info = oversample_vectors(X, y)
    assert info["method_used"] == "smote" and np.bincount(yr).tolist() == [40, 40]
    # pochi esempi: k ridotto o fallback a random, senza errori
    Xr, yr, info = oversample_vectors(X[:42], [1] * 2 + [0] * 40)
    assert np.bincount(yr).tolist() == [40, 40] and info["k_neighbors"] == 1
    Xr, yr, info = oversample_vectors(X[:41], [1] * 1 + [0] * 40)
    assert "fallback" in info["method_used"] and np.bincount(yr).tolist() == [40, 40]


def test_plan_and_augment_with_cache():
    docs = make_docs()
    assert len(plan_synthetic(docs.introverted, 1.0)) == 12                 # raddoppia la minoranza
    assert len(plan_synthetic(docs.introverted, 10.0)) == 36                # tetto: n_maj - n_min
    assert plan_synthetic([1] * 5 + [0] * 5) == []                          # gia' bilanciato
    g, cache = mock_gen(), SyntheticCache()
    syn = augment_minority(docs, "introverted", g, cache=cache, max_chunks=2)
    assert len(syn) == 12 and syn.is_synthetic.all() and (syn.introverted == 1).all()
    assert syn.author.str.contains("#syn").all() and (syn.source_author == syn.author.str.split("#").str[0]).all()
    assert g.calls == 12 * 2                                                # 10 commenti -> 2 blocchi da 5
    syn2 = augment_minority(docs, "introverted", g, cache=cache, max_chunks=2)
    assert g.calls == 24 and list(syn2.text) == list(syn.text)              # seconda volta: tutto dalla cache
    try:
        augment_minority(docs, "introverted", None, cache=SyntheticCache(), max_chunks=2)
        assert False
    except RuntimeError:
        pass
    assert make_chunks(["a", "b", "c"], 2) == ["a b", "c"]


def test_strategy_validation():
    assert set(STRATEGIES) == {"paper_leaky", "noleak_full", "noleak_minimal", "no_oversampling", "undersampling"}
    for bad in (dict(leakage=True, gpt2=False), dict(undersample=True, gpt2=True)):
        kw = dict(name="x", leakage=False, preprocessing="full", gpt2=False, smote=False,
                  undersample=False, description="")
        kw.update(bad)
        try:
            Strategy(**kw)
            assert False
        except ValueError:
            pass


def test_leakage_diagnostic_and_strategies():
    docs = make_docs(n_pos=24, n_neg=96)
    cache = SyntheticCache()
    kw = dict(generator=mock_gen(), cache=cache, features=True, w2v_params=W2V, max_chunks=2, seed=3)

    leaky = prepare_target(docs, "introverted", "paper_leaky", **kw)
    clean = prepare_target(docs, "introverted", "noleak_full", **kw)
    mini = prepare_target(docs, "introverted", "noleak_minimal", **kw)
    nothing = prepare_target(docs, "introverted", "no_oversampling", **kw)
    under = prepare_target(docs, "introverted", "undersampling", **kw)

    # leakage: nel paper la validation contiene righe con la "sorgente" nel train; senza leakage mai
    assert leaky.report["val_rows_with_source_in_train"] > 0
    assert clean.report["val_rows_with_source_in_train"] == 0
    assert mini.report["val_rows_with_source_in_train"] == 0
    assert clean.report["val_synthetic_rows"] == 0 and leaky.report["val_synthetic_rows"] > 0
    # i sintetici entrano solo nel train (senza leakage) e la validation resta sui dati reali
    assert not clean.val.is_synthetic.any() and clean.train.is_synthetic.sum() == clean.report["synthetic_added"]
    assert set(clean.train.author) & set(clean.val.author) == set()
    # SMOTE porta i vettori a classi pari; il testo no
    assert clean.report["train_vectors"]["pos"] == clean.report["train_vectors"]["neg"]
    assert clean.report["train_text_level"]["pos"] != clean.report["train_text_level"]["neg"]
    assert clean.X_train.shape[0] == 2 * clean.report["train_vectors"]["pos"] and clean.X_train.shape[1] == 16
    # senza nulla: distribuzione originale; undersampling: classi pari a livello testo
    assert nothing.report["train_text_level"] == nothing.report["train_before_balancing"]
    assert "smote" not in nothing.report and "synthetic_added" not in nothing.report
    assert under.report["train_text_level"]["pos"] == under.report["train_text_level"]["neg"]
    assert under.val.y.sum() == nothing.val.y.sum()                           # validation non toccata
    # preprocessing diverso -> feature diverse
    assert mini.report["preprocessing"] == "minimal" and clean.report["preprocessing"] == "full"
    # transform per il test: stessa pipeline
    Xt = clean.transform_texts(["cat dog run", "love time home"])
    assert Xt.shape == (2, 16)
    # features=False: solo testo, SMOTE segnalato come non applicato
    only_text = prepare_target(docs, "introverted", "noleak_full", **{**kw, "features": False})
    assert only_text.X_train is None and "non applicato" in only_text.report["smote"]


def test_cache_only_reader_needs_same_signature(tmp_path):
    docs = make_docs()
    db = tmp_path / "c.sqlite"
    g = MockGenerator(lambda t: "gen " + t)
    g.signature = make_signature("gpt2", seed=42)                    # come GPT2Generator di default
    c = SyntheticCache(db)
    syn = augment_minority(docs, "introverted", g, cache=c, max_chunks=2)
    c.close()
    reader = CacheOnlyGenerator(make_signature("gpt2", seed=42))
    syn2 = augment_minority(docs, "introverted", reader, cache=SyntheticCache(db), max_chunks=2)
    assert list(syn2.text) == list(syn.text)                          # letto dalla cache, senza generare
    wrong = CacheOnlyGenerator(make_signature("gpt2", seed=7))
    try:
        augment_minority(docs, "introverted", wrong, cache=SyntheticCache(db), max_chunks=2)
        assert False
    except RuntimeError as e:
        assert "generate_synthetic" in str(e)


def test_gpt2_input_follows_strategy_preprocessing():
    docs = make_docs()
    docs["comments"] = [["Dogs, running! http://x.com I can't stop"] * 5 for _ in range(len(docs))]
    docs["text"] = [" ||| ".join(c) for c in docs.comments]
    seen = []
    g = MockGenerator(lambda t: seen.append(t) or "out " + t)
    fn, tag = make_input_fn("full")
    cache = SyntheticCache()
    augment_minority(docs, "introverted", g, cache=cache, max_chunks=1, input_fn=fn, input_tag=tag)
    assert seen and all("http" not in t and "," not in t and "!" not in t for t in seen)
    assert all(t.startswith("dog run") for t in seen)                       # lemmi, senza punteggiatura
    n = g.calls
    augment_minority(docs, "introverted", g, cache=cache, max_chunks=1, input_fn=fn, input_tag=tag)
    assert g.calls == n                                                      # stessa etichetta: cache
    fn_raw, tag_raw = make_input_fn("raw")
    augment_minority(docs, "introverted", g, cache=cache, max_chunks=1, input_fn=fn_raw, input_tag=tag_raw)
    assert g.calls > n                                                       # input diverso: nuova generazione
    assert any("http" in t for t in seen[n:])
    # prepare_target usa il preprocessing della strategia per l'input di GPT-2
    p_full = prepare_target(docs, "introverted", "noleak_full", generator=mock_gen(), features=False, max_chunks=1)
    p_min = prepare_target(docs, "introverted", "noleak_minimal", generator=mock_gen(), features=False, max_chunks=1)
    assert p_full.report["gpt2_input"] == "full" and p_min.report["gpt2_input"] == "minimal"
    assert prepare_target(docs, "introverted", "no_oversampling", features=False).report["gpt2_input"] is None


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
