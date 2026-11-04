import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.masking import (mask_bigfive_scores, mask_full, mask_limited,
                         load_author_types, process_dataset)


def test_full_mbti():
    out = mask_full("I'm an INTJ, my mom is ENFJs 's friend; Ni-Te and TeNiSeFi. See the Myers-Briggs test.")
    assert "INTJ" not in out and "ENFJs" not in out
    assert "COGFUNC_MENTION" in out and "Myers" not in out
    assert "TeNiSeFi" not in out


def test_full_bigfive_scores_before_vocab():
    out = mask_full("My Big Five: openness 85, conscientiousness: 40%.")
    assert "85" not in out and "40" not in out
    assert "SCORE_MENTION" in out and "openness" not in out.lower()


def test_limited_own_type_only():
    t = "I'm an INTJ but my friends are ENTPs and INFJ."
    out = mask_limited(t, "INTJ", ("mbti",))
    assert "INTJ" not in out and "ENTPs" in out and "INFJ" in out
    out2 = mask_limited("INTJs rule", "INTJ", ("mbti",))
    assert "INTJs" not in out2
    assert mask_limited(t, None, ("mbti",)) == t          # tipo ignoto: nessun masking
    assert mask_limited(t, "INTJ", ("bigfive",)) == t     # regola MBTI non attiva


def test_limited_bigfive_numbers_only():
    t = "Openness: 85, Conscientiousness 40%, I was in the 92nd percentile in neuroticism."
    out = mask_limited(t, None, ("bigfive",))
    assert "Openness" in out and "Conscientiousness" in out and "neuroticism" in out
    for n in ("85", "40", "92"):
        assert n not in out


def test_letters_and_lists_need_context():
    assert "SCORE_MENTION" in mask_bigfive_scores("big five result O: 85 C: 40 E: 22 A: 75 N: 60")
    assert "SCORE_MENTION" in mask_bigfive_scores("OCEAN 85/40/22/75/60")
    for t in ("I wore an N95 mask", "scored in the 90th percentile at school", "A: 5 apples"):
        assert mask_bigfive_scores(t) == t


def test_no_false_positive_plain_text():
    t = "I have 3 cats and 2 dogs, the meeting is at 5 pm."
    assert mask_bigfive_scores(t) == t


def test_pipeline_end_to_end(tmp_path):
    data, out = tmp_path / "d", tmp_path / "o"
    data.mkdir()
    pd.DataFrame({"author": ["a", "b"], "mbti": ["intj", ""]}).to_csv(data / "author_profiles.csv", index=False)
    rows = [
        ("a", "I am an INTJ and my sister is an ENFP, we like long walks outside", "x"),
        ("b", "My scores were openness 85 and conscientiousness 40 on the test", "y"),
        ("a", "[deleted]", "x"),
        ("a", "too short", "x"),
    ]
    pd.DataFrame(rows, columns=["author", "body", "subreddit"]).to_csv(data / "comments_final.csv", index=False)
    types = load_author_types(data / "author_profiles.csv")
    assert types == {"a": "INTJ"}
    stats = process_dataset("final", data, out, ["full", "limited", "none"], types, chunk_size=2)
    assert stats["rows_in"] == 4 and stats["rows_kept"] == 2
    lim = pd.read_csv(out / "limited" / "comments_final.csv", keep_default_na=False)
    assert "INTJ" not in lim.body[0] and "ENFP" in lim.body[0]
    assert "85" not in lim.body[1] and "openness" in lim.body[1]
    full = pd.read_csv(out / "full" / "comments_final.csv", keep_default_na=False)
    assert "ENFP" not in full.body[0] and "openness" not in full.body[1].lower()
    none = pd.read_csv(out / "none" / "comments_final.csv", keep_default_na=False)
    assert "INTJ" in none.body[0]
    assert process_dataset("final", data, out, ["full", "limited", "none"], types) is None


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
