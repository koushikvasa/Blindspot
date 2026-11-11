import json

import pandas as pd
import pyarrow.parquet as pq
import pytest

from blindspot.data import twin2k
from blindspot.data.items import MAX_MISSING, MAX_TOP_SHARE, TWIN_FAMILIES, TWIN_ITEMS, ItemSpec

pytestmark = pytest.mark.skipif(not twin2k.CSV_DIR.exists(), reason="Twin-2K-500 raw data not present")

N_PEOPLE = 2058
DEMO_COLS = ["region", "gender", "age_band", "education", "race", "income_band"]


@pytest.fixture(scope="module")
def data():
    return twin2k.load()


def test_three_files_with_expected_shapes(data):
    assert set(data) == {"real", "gpt41mini_twin", "retest"}
    for r in data.values():
        assert r.answers.shape == (N_PEOPLE, len(TWIN_ITEMS))
        assert list(r.demographics.columns) == DEMO_COLS


def test_wording_row_is_dropped(data):
    ids = data["real"].answers.index
    assert "TWIN_ID" not in ids
    assert ids.str.fullmatch(r"\d+").all()
    assert ids.is_unique


def test_files_share_ids_items_and_demographics(data):
    real = data["real"]
    for r in data.values():
        assert r.answers.index.equals(real.answers.index)
        assert r.items == real.items
        assert r.demographics.equals(real.demographics)


def test_answers_are_option_indices_or_nan(data):
    for r in data.values():
        for it in r.items:
            vals = r.answers[it.code].dropna()
            assert vals.between(0, len(it.options) - 1).all()
            assert (vals % 1 == 0).all()


def test_items_follow_selection_rules_in_real(data):
    a = data["real"].answers
    assert (a.isna().mean() < MAX_MISSING).all()
    assert a.apply(lambda s: s.value_counts(normalize=True).max()).le(MAX_TOP_SHARE).all()


def test_families_flags_and_wording(data):
    items = data["real"].items
    assert {i.family for i in items} == set(TWIN_FAMILIES)
    assert {i.code for i in items if i.sensitive} == {
        i.code for i in items if i.family in ("policy_self", "perception")}
    assert all(i.wording and i.verified for i in items)
    assert sum(i.live_ok for i in items) == 10


def test_text_and_float_codes_map_to_the_same_index():
    spec = ItemSpec("x", "x", "t", "ordinal")
    assert twin2k.encode(pd.Series(["3", "3.0", None]), spec, 5).tolist()[:2] == [2.0, 2.0]
    with pytest.raises(ValueError):
        twin2k.encode(pd.Series(["6"]), spec, 5)


def test_percent_bins_edges():
    spec = next(s for s in TWIN_ITEMS if s.bins)
    got = twin2k.encode(pd.Series(["0", "19.0", "20", "59", "60", "80", "100"]), spec, 5).tolist()
    assert got == [0, 0, 1, 2, 3, 4, 4]


def test_csv_codes_match_persona_labels(data):
    """persona_json holds wave-4 answers with the same 1..k coding: retest index == SelectedByPosition - 1."""
    rows = pq.read_table(sorted(twin2k.PERSONA_DIR.glob("*.parquet"))[0], columns=["pid", "persona_json"]).to_pylist()
    retest = data["retest"].answers
    policy = [s.code for s in TWIN_ITEMS if s.theme == "policy_self"]
    checked = 0
    for row in rows:
        for q in twin2k._questions(row["persona_json"]):
            if q["QuestionID"] == "QID287":
                expected = [p - 1 for p in q["Answers"]["SelectedByPosition"]]
                assert retest.loc[str(row["pid"]), policy].tolist() == expected
                checked += 1
            elif q["QuestionID"] == "QID291":
                assert retest.loc[str(row["pid"]), "omission_vaccine"] == q["Answers"]["SelectedByPosition"] - 1
    assert checked == len(rows)


def test_demographics_come_only_from_demographics_block(data):
    demo = data["real"].demographics
    assert demo.notna().all().all()
    assert set(demo["age_band"]) <= {"18-29", "30-49", "50-64", "65+"}
    assert not any("persona" in c for c in demo.columns)
    assert all(len(json.dumps(v)) < 200 for v in demo.iloc[0].tolist())
