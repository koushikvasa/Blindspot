"""Exported UI files (results/ui): privacy scan and internal consistency. Skipped until the export has run."""
import json
import re
from pathlib import Path

import pytest

UI = Path(__file__).resolve().parents[1] / "results" / "ui"
pytestmark = pytest.mark.skipif(not UI.exists(), reason="results/ui not exported yet")
FORBIDDEN_KEYS = {"id", "ids", "respondent_id", "uncovered_ids", "asked", "asked_new", "persona", "personas", "reason",
                  "reasons", "segment", "segments", "demographic_reason"}
PERSONA_WORDS = re.compile(r"\bpersonas?\b|you are a |respondent id|wave[- ]?4", re.I)


def files() -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(UI.glob("*.json"))}


def walk(node, path=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield f"{path}/{k}", k, v
            yield from walk(v, f"{path}/{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from walk(v, f"{path}[{i}]")


def test_no_ids_persona_text_or_demographic_reasons_in_any_ui_file():
    for name, data in files().items():
        for where, key, value in walk(data):
            assert key not in FORBIDDEN_KEYS, (name, where)
            if isinstance(value, str):
                assert not PERSONA_WORDS.search(value), (name, where, value[:80])
                assert not re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-.*", value), (name, where)


def test_expected_files_exist_and_real_realrecruit_file_is_absent():
    names = set(files())
    assert {"summary", "repairs_gss", "scoreboard_gss", "fairness_gss", "live_gss"} <= names
    assert "realrecruit_gss" not in names


def test_maps_are_in_unit_square_with_one_fixed_real_layout_per_dataset():
    f = files()
    for ds in ("twin", "gss"):
        maps = {n: d for n, d in f.items() if n.startswith(f"map_{ds}_")}
        assert maps, ds
        reals = [[(x, y) for x, y, _ in d["real"]] for d in maps.values()]
        assert all(r == reals[0] for r in reals), f"{ds}: real layout differs between crowds"
        for d in maps.values():
            assert all(0 <= v <= 1 for x, y, _ in d["real"] for v in (x, y))
            assert all(0 <= v <= 1 for x, y in d["synthetic"] for v in (x, y))


def test_every_summary_crowd_has_a_map_whose_covered_share_matches_its_coverage():
    f = files()
    for ds, block in f["summary"]["datasets"].items():
        for crowd, entry in block["crowds"].items():
            m = f[f"map_{ds}_{crowd}"]
            share = sum(c for _, _, c in m["real"]) / len(m["real"])
            assert round(share, 3) == entry["coverage"], (ds, crowd)


def test_holdout_coverage_is_present_only_where_a_holdout_number_exists_and_is_a_probability():
    for block in files()["summary"]["datasets"].values():
        for entry in block["crowds"].values():
            if "holdout_coverage" in entry:
                assert 0 <= entry["holdout_coverage"] <= 1


def test_live_bundle_is_consistent_and_small_groups_are_excluded_from_fairness():
    f = files()
    live = f["live_gss"]
    k = len(live["items"])
    assert all(len(r) == k for r in live["real"]) and all(len(r) == k for r in live["crowd"])
    assert len(live["real"]) == len(live["real_xy"])
    assert all(g["n"] >= 30 for g in f["fairness_gss"]["groups"])
    sb = f["scoreboard_gss"]
    for m in sb["methods"].values():
        assert all(lo <= mean <= hi for lo, mean, hi in zip(m["lo"], m["mean"], m["hi"]))
