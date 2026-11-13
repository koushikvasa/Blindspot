import json

import numpy as np
import pandas as pd
import pytest

from blindspot.data.schema import Item, Respondents
from blindspot.eval.experiment0 import DatasetInput, HoldoutError, run
from blindspot.eval.splits import make_splits

N = 200


def _toy_dataset() -> tuple[Respondents, dict, Respondents]:
    rng = np.random.default_rng(7)
    items = [Item(f"q{i}", "?", "ordinal", list("abcd"), "f1" if i < 3 else "f2") for i in range(5)]
    ids = [str(i) for i in range(N)]
    ans = pd.DataFrame(rng.integers(0, 4, size=(N, 5)), index=ids, columns=[i.code for i in items], dtype=float)
    demo = pd.DataFrame({"age_band": rng.choice(["18-29", "30-49"], N)}, index=ids)
    real = Respondents(ans, demo, items, "toy:real")
    splits = make_splits(ids, seed=42)
    twin = Respondents(ans.clip(1, 2), demo, items, "toy:twin")  # a narrowed copy of everyone
    return real, splits, twin


def test_experiment0_runs_end_to_end_on_toy_data(tmp_path):
    real, splits, twin = _toy_dataset()
    keep = splits["pool"] + splits["test"]
    d = DatasetInput("toy", "Toy", real.subset(keep), splits,
                     fixed_crowds={"narrow_twin": twin.subset(splits["test"])})
    report = run([d], out_dir=tmp_path, n_boot=20, maps=False)

    crowds = report["datasets"]["toy"]["crowds"]
    assert set(crowds) == {"narrow_twin", "independent_marginals", "gaussian_copula", "other_real_people"}
    for m in crowds.values():
        assert 0 <= m["coverage"] <= 1 and m["coverage_ci"][0] <= m["coverage"] <= m["coverage_ci"][1]
        assert m["n_synthetic"] == len(splits["test"])
        assert "uncovered_ids" not in m and 0 <= m["n_uncovered"] <= len(splits["test"])
    saved = json.loads((tmp_path / "experiment0.json").read_text(encoding="utf-8"))
    assert saved["settings"]["k"] == 5 and "git_commit" in saved["settings"]
    md = (tmp_path / "experiment0.md").read_text(encoding="utf-8")
    assert "| crowd | n | coverage [95% CI]" in md and "Per-family" in md and "k=10" in md
    assert not any(h in json.dumps(saved) for h in [f'"{i}"' for i in splits["holdout"]])


def test_experiment0_refuses_holdout_people(tmp_path):
    real, splits, _ = _toy_dataset()
    d = DatasetInput("toy", "Toy", real, splits)  # holdout rows not removed
    with pytest.raises(HoldoutError):
        run([d], out_dir=tmp_path, n_boot=5, maps=False)
