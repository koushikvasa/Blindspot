"""Real-respondent loader and experiment on a FAKE fixture (no GSS data, no network, no Claude)."""
import numpy as np
import pandas as pd
import pytest

from blindspot.data.schema import Item, Respondents
from blindspot.eval.experiment_real import effective_n, evaluate_recruits, rake, to_markdown, weighted_copula
from blindspot.coverage import real_distances
from blindspot.generators.flashpoint_real import SOURCE, load

OPTS = ["a", "b", "c"]
ITEMS = [Item(f"q{i}", "w", "ordinal", OPTS, "fam1" if i < 3 else "fam2") for i in range(6)]
AGE, GEN = ["18-29", "30-49", "50-64", "65+"], ["male", "female"]
EDU = ["less than high school", "high school", "associate/junior college", "bachelor's", "graduate"]
INC = ["Less than $30,000", "$30,000-$49,999", "$50,000-$74,999", "$75,000-$109,999", "$110,000 or more"]


def fixture_csv(path, n=12, drop_status=(), excluded=()):
    rng = np.random.default_rng(0)
    rows = {"#": [str(i) for i in range(n)], "ID": [f"x{i}" for i in range(n)],
            "Status": ["COMPLETE"] * n, "Quality Excluded": [None] * n}
    for i in drop_status:
        rows["Status"][i] = "INCOMPLETE"
    for i in excluded:
        rows["Quality Excluded"][i] = "true"
    for it in ITEMS:
        rows[it.code] = [OPTS[j] for j in rng.integers(0, 3, n)]
    rows["age_band"] = [AGE[i % 4] for i in range(n)]
    rows["gender"] = [GEN[i % 2] for i in range(n)]
    rows["education"] = [EDU[i % 5] for i in range(n)]
    rows["income_band"] = [INC[i % 5] for i in range(n)]
    df = pd.DataFrame(rows)
    labels = pd.DataFrame([{c: "#" if c == "#" else f"text {c}" for c in df.columns}])
    pd.concat([df.iloc[:0], labels, df]).to_csv(path, index=False)      # second header row like the real export
    return path


def test_loader_keeps_complete_rows_drops_excluded_and_maps_everything(tmp_path):
    csv = fixture_csv(tmp_path / "r.csv", n=14, drop_status=(0,), excluded=(1,))
    resp, rep = load(csv, ITEMS)
    assert resp.source == SOURCE == "flashpoint:prolific_real"
    assert (rep.n_rows_in_file, rep.n_complete, rep.n_quality_excluded, rep.n_kept) == (14, 13, 1, 12)
    assert rep.answers.n_invalid == 0 and not rep.demographics_unmapped
    assert list(resp.answers.index)[:2] == ["fp_real_0000", "fp_real_0001"]
    assert set(resp.demographics["education"]) <= set(EDU)
    assert (resp.demographics["region"] == "unknown").all()


def test_loader_refuses_fewer_than_ten_usable_people(tmp_path):
    with pytest.raises(ValueError, match="need at least 10"):
        load(fixture_csv(tmp_path / "r.csv", n=9), ITEMS)


def test_loader_flags_unmapped_demographics(tmp_path):
    csv = fixture_csv(tmp_path / "r.csv", n=12)
    df = pd.read_csv(csv, dtype=str)
    df.loc[2, "gender"] = "nonbinary"
    df.to_csv(csv, index=False)
    _, rep = load(csv, ITEMS)
    assert rep.demographics_unmapped == {"gender": {"nonbinary": 1}}


def test_rake_hits_target_margins_and_effective_n_is_smaller():
    demo = pd.DataFrame({"gender": ["male"] * 8 + ["female"] * 2, "age_band": ["18-29", "30-49"] * 5})
    targets = {"gender": pd.Series({"male": 0.5, "female": 0.5}), "age_band": pd.Series({"18-29": 0.5, "30-49": 0.5})}
    w = rake(demo, targets, trim=(0.0, 100.0))
    male = w[(demo.gender == "male").to_numpy()].sum() / w.sum()
    assert male == pytest.approx(0.5, abs=1e-6)
    assert effective_n(w) < len(w)
    assert w.mean() == pytest.approx(1.0)


def synthetic_study(seed=0):
    rng = np.random.default_rng(seed)

    def people(prefix, n):
        idx = pd.Index([f"{prefix}{i}" for i in range(n)])
        demo = pd.DataFrame({"age_band": rng.choice(AGE, n), "gender": rng.choice(GEN, n),
                             "education": rng.choice(EDU, n), "income_band": rng.choice(INC, n),
                             "region": "south", "race": "white", "weight": 1.0}, index=idx)
        return Respondents(pd.DataFrame(rng.integers(0, 3, (n, 6)).astype(float), index=idx,
                                        columns=[i.code for i in ITEMS]), demo, ITEMS, "synthetic")
    return people("t", 60), people("p", 80), people("b", 60)


def test_experiment_runs_on_a_partial_fake_file(tmp_path):
    target, pool, base = synthetic_study()
    splits = {"pool": list(pool.answers.index), "test": list(target.answers.index), "holdout": ["h0"]}
    recruits, _ = load(fixture_csv(tmp_path / "r.csv", n=12), ITEMS)
    res = evaluate_recruits(target, pool, base, recruits, splits, real_distances(target, target.items), n_boot=10)
    assert set(res["crowds"]) == {"claude_mode", "copula_on_recruits", "copula_raked", "claude_plus_recruits"}
    for m in res["crowds"].values():
        assert 0.0 <= m["coverage"] <= 1.0 and 0.0 <= m["precision"] <= 1.0
    assert res["crowds"]["claude_plus_recruits"]["near_dup_vs_recruits"] is not None
    assert 0 < res["raking"]["effective_n"] <= 12
    report = {"settings": {"n_recruits": 12, "partial": True, "n_items": 6, "n_target": 60, "k": 5, "n_boot": 10,
                           "git_commit": "x", "weight_trim": [0.2, 5.0]},
              "load": {"n_rows_in_file": 13, "n_complete": 12, "n_quality_excluded": 0,
                       "answers": {"n_answers": 72, "missing": 0, "unmapped": 0, "invalid_share": 0.0},
                       "demographics_unmapped": {}}, "runtime_seconds": 1.0, **res}
    assert "PARTIAL" in to_markdown(report)


def test_weighted_copula_with_equal_weights_returns_valid_options():
    target, pool, _ = synthetic_study()
    crowd = weighted_copula(pool, target.demographics, np.ones(len(pool.answers)), seed=1)
    assert crowd.answers.shape == (60, 6)
    assert set(np.unique(crowd.answers.to_numpy())) <= {0.0, 1.0, 2.0}
