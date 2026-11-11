import numpy as np
import pandas as pd
import pytest

from blindspot.data.schema import Item, Respondents
from blindspot.space import encode, family_weights, gower


def _items(families):
    return [Item(f"q{i}", f"Q{i}?", "ordinal", ["a", "b", "c"], fam) for i, fam in enumerate(families)]


def test_encode_scales_to_unit_interval_and_keeps_nan():
    items = [Item("a", "A?", "ordinal", ["x", "y", "z"], "t"), Item("b", "B?", "binary", ["n", "y"], "t")]
    ans = pd.DataFrame({"a": [0, 2, np.nan], "b": [1, 0, 1]}, index=["1", "2", "3"], dtype=float)
    x = encode(Respondents(ans, pd.DataFrame(index=ans.index), items, "t"))
    assert x[0].tolist() == [0.0, 1.0] and x[1].tolist() == [1.0, 0.0]
    assert np.isnan(x[2, 0]) and x[2, 1] == 1.0


def test_identical_people_have_zero_distance_and_opposites_one():
    items = _items(["f1", "f1", "f2"])
    a = np.array([[0.0, 0.5, 1.0], [0.0, 0.0, 0.0]])
    b = np.array([[0.0, 0.5, 1.0], [1.0, 1.0, 1.0]])
    d = gower(a, b, items)
    assert d[0, 0] == pytest.approx(0.0)
    assert d[1, 1] == pytest.approx(1.0)


def test_family_weighting_gives_each_family_equal_weight():
    items = _items(["big", "big", "big", "small"])
    assert family_weights(items).tolist() == pytest.approx([1 / 6, 1 / 6, 1 / 6, 1 / 2])
    a = np.array([[0.0, 0.0, 0.0, 0.0]])
    differs_in_small = np.array([[0.0, 0.0, 0.0, 1.0]])
    differs_in_one_big = np.array([[1.0, 0.0, 0.0, 0.0]])
    assert gower(a, differs_in_small, items)[0, 0] == pytest.approx(0.5)
    assert gower(a, differs_in_one_big, items)[0, 0] == pytest.approx(1 / 6)


def test_nan_items_are_skipped_and_weights_renormalized():
    items = _items(["f1", "f2"])
    a = np.array([[0.0, np.nan]])
    b = np.array([[1.0, 0.0]])
    assert gower(a, b, items)[0, 0] == pytest.approx(1.0)
    assert np.isnan(gower(np.array([[np.nan, np.nan]]), b, items)[0, 0])


def test_chunking_does_not_change_result():
    rng = np.random.default_rng(0)
    items = _items(["a", "a", "b", "c", "c"])
    x = rng.random((50, 5))
    x[rng.random((50, 5)) < 0.1] = np.nan
    assert np.allclose(gower(x, x, items, chunk=7), gower(x, x, items, chunk=1000), equal_nan=True)
