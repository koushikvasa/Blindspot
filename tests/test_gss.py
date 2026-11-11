import pandas as pd
import pyreadstat
import pytest

from blindspot.data import gss
from blindspot.data.items import GSS_FAMILIES, GSS_ITEMS, MAX_MISSING, MAX_TOP_SHARE, ItemSpec

pytestmark = pytest.mark.skipif(not gss.DTA.exists(), reason="GSS 2024 raw data not present")

UNVERIFIED = {"trppl", "trcourts", "trcong", "intcnfrm", "satdemoc",
              "techesy", "harmgood1", "aiworry", "aidrive", "aimed"}
SENSITIVE = {"trcong", "satdemoc", "polint", "tax"}


@pytest.fixture(scope="module")
def r():
    return gss.load()


def test_target_set_is_main_cross_section_ballot_1(r):
    df, _ = pyreadstat.read_dta(str(gss.DTA), usecols=["id", "sample", "ballot"])
    expected = df[(df["sample"] == 13) & (pd.to_numeric(df["ballot"]) == 1)]["id"].astype(int).astype(str)
    assert len(r.answers) == 1116
    assert set(r.answers.index) == set(expected)


def test_shapes_and_weight(r):
    assert r.answers.shape == (1116, 24)
    assert r.demographics["weight"].notna().all()
    assert {"age_band", "gender", "education", "income_band", "region", "race"} <= set(r.demographics)


def test_items_follow_selection_rules(r):
    a = r.answers
    assert (a.isna().mean() < MAX_MISSING).all()
    assert a.apply(lambda s: s.value_counts(normalize=True).max()).le(MAX_TOP_SHARE).all()
    for it in r.items:
        vals = a[it.code].dropna()
        assert vals.between(0, len(it.options) - 1).all()


def test_four_themes_of_six_with_flags(r):
    fams = pd.Series([i.family for i in r.items]).value_counts()
    assert set(fams.index) == set(GSS_FAMILIES) and (fams == 6).all()
    assert {i.code for i in r.items if not i.verified} == UNVERIFIED
    assert {i.code for i in r.items if i.sensitive} == SENSITIVE
    assert {i.code for i in r.items if i.live_ok} == {i.code for i in r.items} - UNVERIFIED - SENSITIVE


def test_option_labels_from_value_labels(r):
    by = {i.code: i for i in r.items}
    assert by["satfin"].options == ["pretty well satisfied", "more or less satisfied", "not satisfied at all"]
    assert by["finalter"].options == ["better", "stayed same", "worse"]
    assert by["trcourts"].options[0] == "0 - no trust" and len(by["trcourts"].options) == 11


def test_encode_maps_codes_by_position_and_rejects_unknown():
    spec = ItemSpec("finalter", "finalter", "t", "ordinal")
    got = gss.encode(pd.Series([1, 2, 3, None]), spec, (1, 3, 2)).tolist()
    assert got[:3] == [0.0, 2.0, 1.0] and pd.isna(got[3])
    with pytest.raises(ValueError):
        gss.encode(pd.Series([4]), spec, (1, 3, 2))


def test_every_spec_has_wording():
    from blindspot.data.items import GSS_WORDING
    assert {s.code for s, _ in GSS_ITEMS} == set(GSS_WORDING)
