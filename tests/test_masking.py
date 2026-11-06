import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.masking import (mask_bigfive_scores, mask_full, mask_limited,
                         load_author_types, process_dataset)
from src.masking.rules import load_excluded_subreddits


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
    pd.DataFrame({"author": ["a", "b"], "mbti": ["intj", ""],
                  "openness": ["80", "None"], "conscientiousness": ["10", "20"],
                  "extraversion": ["30", "40"], "agreeableness": ["50", "60"],
                  "neuroticism": ["70", "80"]}).to_csv(data / "author_profiles.csv", index=False)
    rows = [
        ("a", "I am an INTJ and my sister is an ENFP, we like long walks outside", "x"),
        ("b", "My scores were openness 85 and conscientiousness 40 on the test", "y"),
        ("a", "[deleted]", "x"),
        ("a", "too short", "x"),
    ]
    pd.DataFrame(rows, columns=["author", "body", "subreddit"]).to_csv(data / "comments_final.csv", index=False)
    types = load_author_types(data / "author_profiles.csv", "mbti")
    assert types == {"a": {"mbti": "INTJ"}}
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


def test_load_excluded_subreddits(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("5,entp\n7,MBTI\n\n9,r/INTJ\n# commento\n", encoding="utf-8")
    assert load_excluded_subreddits(p) == {"entp", "mbti", "intj"}


def test_full_drops_excluded_subreddits_only(tmp_path):
    data, out = tmp_path / "d", tmp_path / "o"
    data.mkdir()
    pd.DataFrame({"author": ["a"], "mbti": ["intj"]}).to_csv(data / "author_profiles.csv", index=False)
    rows = [
        ("a", "I really like talking about my own personality type here", "INTJ"),
        ("a", "Weather was great and we went hiking for several hours", "hiking"),
        ("a", "Another long comment about cooking pasta for the whole family", "r/Cooking"),
        ("a", "Posting this one in the mbti subreddit where everyone talks", "mbti"),
    ]
    pd.DataFrame(rows, columns=["author", "body", "subreddit"]).to_csv(data / "comments_final.csv", index=False)
    types = load_author_types(data / "author_profiles.csv", "mbti")
    st = process_dataset("final", data, out, ["none", "limited", "full"], types,
                         chunk_size=3, excluded_subreddits={"intj", "mbti"})
    full = pd.read_csv(out / "full" / "comments_final.csv", keep_default_na=False)
    none = pd.read_csv(out / "none" / "comments_final.csv", keep_default_na=False)
    lim = pd.read_csv(out / "limited" / "comments_final.csv", keep_default_na=False)
    assert list(full.subreddit) == ["hiking", "r/Cooking"]          # r/INTJ e r/mbti scartati (anche con maiuscole)
    assert len(none) == 4 and len(lim) == 4                          # gli altri livelli tengono tutto
    assert st["rows_out_by_level"] == {"none": 4, "limited": 4, "full": 2}
    assert st["dropped_excluded_subreddit_by_level"]["full"] == 2
    # senza lista, full tiene tutte le righe
    out2 = tmp_path / "o2"
    st2 = process_dataset("final", data, out2, ["full"], types)
    assert st2["rows_out_by_level"]["full"] == 4


def test_full_all_rows_excluded_gives_header_only_file(tmp_path):
    data, out = tmp_path / "d", tmp_path / "o"
    data.mkdir()
    pd.DataFrame({"author": ["a"], "mbti": ["intj"]}).to_csv(data / "author_profiles.csv", index=False)
    pd.DataFrame([("a", "A perfectly fine long enough comment for the filter", "intj")],
                 columns=["author", "body", "subreddit"]).to_csv(data / "comments_final.csv", index=False)
    process_dataset("final", data, out, ["full"], {}, excluded_subreddits={"intj"})
    res = pd.read_csv(out / "full" / "comments_final.csv", keep_default_na=False)
    assert len(res) == 0 and list(res.columns) == ["author", "body", "subreddit"]


def _profiles(path):
    pd.DataFrame({
        "author": ["ok", "null_trait", "none_str", "bad_type", "dup", "dup"],
        "mbti": ["intj", "enfp", "infj", "xxxx", "entp", ""],
        "openness": ["85", "", "60", "50", "10", "20"],
        "conscientiousness": ["40", "40", "None", "50", "11", "21"],
        "extraversion": ["22", "22", "22", "50", "12", "22"],
        "agreeableness": ["75", "75", "75", "50", "13", "23"],
        "neuroticism": ["60", "60", "60", "50", "14", "24"],
    }).to_csv(path, index=False)


def test_load_author_types_mbti_unchanged(tmp_path):
    _profiles(tmp_path / "p.csv")
    got = load_author_types(tmp_path / "p.csv", "mbti")
    assert got["ok"] == {"mbti": "INTJ"} and "bad_type" not in got
    assert got["null_trait"] == {"mbti": "ENFP"}          # i Big Five non contano per l'MBTI
    assert got["dup"] == {"mbti": "ENTP"}                 # la riga col tipo valido


def test_load_author_types_big_five_not_null(tmp_path):
    _profiles(tmp_path / "p.csv")
    got = load_author_types(tmp_path / "p.csv", "big_five")
    assert set(got) == {"ok", "bad_type", "dup"}           # celle vuote e "None" escluse; il tipo MBTI non conta
    assert got["ok"] == {"openness": 85.0, "conscientiousness": 40.0, "extraversion": 22.0,
                         "agreeableness": 75.0, "neuroticism": 60.0}
    assert all(isinstance(v, float) for lab in got.values() for v in lab.values())   # sempre definiti


def test_load_author_types_final_needs_both(tmp_path):
    _profiles(tmp_path / "p.csv")
    got = load_author_types(tmp_path / "p.csv", "final")
    assert set(got) == {"ok", "dup"}
    assert "null_trait" not in got and "none_str" not in got and "bad_type" not in got
    assert got["ok"]["mbti"] == "INTJ" and got["ok"]["openness"] == 85.0


def test_load_author_types_errors(tmp_path):
    pd.DataFrame({"author": ["a"], "mbti": ["intj"]}).to_csv(tmp_path / "p.csv", index=False)
    try:
        load_author_types(tmp_path / "p.csv", "big_five")
        assert False, "doveva dare errore: mancano le colonne Big Five"
    except ValueError as e:
        assert "openness" in str(e)
    try:
        load_author_types(tmp_path / "p.csv", "boh")
        assert False
    except ValueError:
        pass


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
