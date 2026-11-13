"""Flashpoint loader: label -> option index mapping, invalid-answer accounting, ids and source."""
import pandas as pd
import pytest

from blindspot.data.schema import Item
from blindspot.generators.flashpoint_panel import SOURCE, load, map_answers, normalize

ITEMS = [
    Item("happy", "q", "ordinal", ["very happy", "pretty happy", "not too happy"], "work_life"),
    Item("trppl", "q", "ordinal", ["0 - you can't be too careful"] + [str(i) for i in range(1, 10)]
         + ["10 - most people can be trusted"], "trust"),
]


def test_normalize_straightens_quotes_case_and_spaces():
    assert normalize("  Very   Happy. ") == "very happy"
    assert normalize("0 - you can’t be too careful") == "0 - you can't be too careful"


def test_maps_labels_and_bare_numbers_to_indices():
    raw = pd.DataFrame({"happy": ["very happy", "Not too happy"], "trppl": ["7", "10 - most people can be trusted"]})
    answers, report = map_answers(raw, ITEMS)
    assert answers["happy"].tolist() == [0.0, 2.0]
    assert answers["trppl"].tolist() == [7.0, 10.0]
    assert report.n_invalid == 0


def test_counts_unmapped_and_missing_without_guessing():
    raw = pd.DataFrame({"happy": ["very happy", "kinda", None], "trppl": ["11", "5", ""]})
    answers, report = map_answers(raw, ITEMS)
    assert answers["happy"].isna().tolist() == [False, True, True]
    assert report.unmapped == {"happy": {"kinda": 1}, "trppl": {"11": 1}}
    assert report.missing == {"happy": 1, "trppl": 1}
    assert report.n_invalid == 4


def test_missing_item_column_raises():
    with pytest.raises(ValueError, match="happy"):
        map_answers(pd.DataFrame({"trppl": ["1"]}), ITEMS)


def test_load_sets_source_ids_and_unknown_demographics(tmp_path):
    path = tmp_path / "rows.csv"
    pd.DataFrame({"happy": ["pretty happy"], "trppl": ["3"]}).to_csv(path, index=False)
    resp, report = load(path, ITEMS, "q9")
    assert resp.source == SOURCE == "flashpoint:uncalibrated"
    assert list(resp.answers.index) == ["fp_q9_0000"]
    assert resp.demographics.loc["fp_q9_0000", "gender"] == "unknown"
    assert report.n_invalid == 0


def test_load_uses_persona_columns_when_present(tmp_path):
    path = tmp_path / "rows.csv"
    pd.DataFrame({"happy": ["pretty happy"], "trppl": ["3"], "age": ["34"], "gender": ["Female"],
                  "education": ["High School"], "income_bracket": ["25-50k"], "ethnicity": ["Hispanic"]}
                 ).to_csv(path, index=False)
    resp, _ = load(path, ITEMS, "q9")
    row = resp.demographics.iloc[0]
    assert (row["age_band"], row["gender"], row["education"]) == ("30-49", "female", "high school")
