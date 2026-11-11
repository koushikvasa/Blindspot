import pandas as pd
import pytest

from blindspot.data.schema import Item, Respondents


def test_respondents_shape_checks():
    items = [Item("q1", "Q1?", "binary", ["no", "yes"], "trust")]
    ans = pd.DataFrame({"q1": [0, 1]}, index=["a", "b"])
    demo = pd.DataFrame({"age_band": ["18-29", "30-44"]}, index=["a", "b"])
    r = Respondents(ans, demo, items, "test")
    assert len(r.answers) == 2
    with pytest.raises(AssertionError):
        Respondents(ans, demo.iloc[:1], items, "test")
