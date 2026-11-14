"""Audit feature: parsing, matching, estimator, privacy (nothing on disk, nothing logged), limits, rate limit, sample flow."""
import csv
import io
import logging
import tempfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

from blindspot.api import main
from blindspot.audit import api as audit_api
from blindspot.audit.engine import Estimator, expected_error, load_calibration, prepare
from blindspot.audit.match import match_columns
from blindspot.audit.parse import AuditError, parse_csv
from blindspot.audit.report import build_report

client = TestClient(main.app)
LIKERT = ["strongly disagree", "disagree", "neutral", "agree", "strongly agree"]


@pytest.fixture(autouse=True)
def fresh_rate_limit():
    audit_api._hits.clear()
    yield
    audit_api._hits.clear()


def to_csv(headers, rows) -> str:
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(headers)
    w.writerows(rows)
    return out.getvalue()


def likert_rows(n, seed, centre=None, spread=1.2, cols=6):
    rng = np.random.default_rng(seed)
    base = rng.normal(2 if centre is None else centre, 1.4, size=(n, 1))
    vals = np.clip(np.rint(base + rng.normal(0, spread, size=(n, cols))), 0, 4).astype(int)
    return [[LIKERT[v] for v in row] for row in vals]


HEADERS = [f"Q{j}" for j in range(6)]


# ---- parsing
def test_parse_flashpoint_style_export_drops_second_header_row():
    text = to_csv(["#", "Q1", "Q2"], [["#", "Q1", "Q2"], ["1", "agree", "yes"], ["2", "disagree", "no"]])
    t = parse_csv(text, "x")
    assert t.headers == ("#", "Q1", "Q2") and t.n_rows == 2


def test_parse_numeric_likert_and_semicolons_and_bom():
    t = parse_csv("﻿a;b\n1;5\n2;4\n3;3\n", "x")
    assert t.headers == ("a", "b") and t.rows[0] == ("1", "5")


def test_parse_messy_headers_are_deduped_and_blank_named():
    t = parse_csv("a,a,\n1,2,3\n", "x")
    assert t.headers == ("a", "a (2)", "column 3")


@pytest.mark.parametrize("text", ["", "   \n", "only,a,header\n", "a,b\n\x00\x01\n1,2\n"])
def test_parse_rejects_bad_files(text):
    with pytest.raises(AuditError):
        parse_csv(text, "x")


def test_parse_limits():
    with pytest.raises(AuditError) as e:
        parse_csv("a\n" + "1\n" * 5001, "x")
    assert e.value.status == 413
    with pytest.raises(AuditError) as e:
        parse_csv(",".join(f"c{i}" for i in range(101)) + "\n" + ",".join("1" * 101) + "\n", "x")
    assert e.value.status == 413
    with pytest.raises(AuditError) as e:
        parse_csv("a\n" + "x" * (5 * 1024 * 1024 + 10), "x")
    assert e.value.status == 413


# ---- matching
def test_match_columns_exact_normalized_and_fuzzy():
    matches, un_panel, un_real = match_columns(("Trust People", "income_level", "extra"), ("trust_people", "Income level", "only real"))
    methods = {m.real: m.method for m in matches}
    assert methods["trust_people"] == "normalized" and methods["Income level"] == "normalized"
    assert un_panel == ["extra"] and un_real == ["only real"]
    matches, _, _ = match_columns(("satisfaction_with_life",), ("satisfaction_with_lyfe",))
    assert matches[0].method == "fuzzy"


def test_metadata_columns_are_ignored_by_default():
    real = to_csv(["respondent id", *HEADERS], [[i, *r] for i, r in enumerate(likert_rows(30, 1))])
    panel = to_csv(["respondent id", *HEADERS], [[i, *r] for i, r in enumerate(likert_rows(60, 2))])
    prep = prepare(parse_csv(panel, "p"), parse_csv(real, "r"))
    assert "respondent id" not in [c.name for c in prep.columns]
    assert any(x["name"] == "respondent id" and x["reason"] for x in prep.entries)


# ---- estimator
def test_estimator_separates_matching_and_mismatched_crowds():
    real = likert_rows(120, 10)
    good = prepare(parse_csv(to_csv(HEADERS, likert_rows(300, 11)), "p"), parse_csv(to_csv(HEADERS, real), "r"))
    bad = prepare(parse_csv(to_csv(HEADERS, [["strongly agree"] * 6] * 300), "p"), parse_csv(to_csv(HEADERS, real), "r"))
    missed_good = build_report(good)["summary"]["missed"]
    missed_bad = build_report(bad)["summary"]["missed"]
    assert missed_bad > missed_good + 0.3
    assert missed_good < 0.35


def test_expected_error_table_and_reliable_flag():
    cal = load_calibration()
    assert expected_error(25)["reliable"] is False
    assert expected_error(200)["reliable"] is True
    assert expected_error(25)["rmse"] > expected_error(200)["rmse"]
    assert isinstance(Estimator(**cal["estimator"]), Estimator)


def test_too_few_real_rows_is_a_clear_error():
    panel, real = to_csv(HEADERS, likert_rows(50, 1)), to_csv(HEADERS, likert_rows(5, 2))
    r = client.post("/api/audit/run", json={"panel": panel, "real": real})
    assert r.status_code == 422 and "at least" in r.json()["detail"]


def test_no_matching_columns_is_a_clear_error():
    r = client.post("/api/audit/preview", json={"panel": to_csv(["a", "b"], [[1, 2]] * 3), "real": to_csv(["x", "y"], [[1, 2]] * 30)})
    assert r.status_code == 422 and "match" in r.json()["detail"]


# ---- privacy
def test_run_writes_nothing_to_disk_and_logs_no_content(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    secret = "zebra-quokka-9000"
    panel = to_csv(["Q0", "Q1", "Q2"], [[secret, "agree", "yes"]] * 60)
    real = to_csv(["Q0", "Q1", "Q2"], [[secret, LIKERT[i % 5], "yes" if i % 2 else "no"] for i in range(40)])
    with caplog.at_level(logging.DEBUG):
        for path in ("preview", "run", "report.html", "repaired.csv"):
            r = client.post(f"/api/audit/{path}", json={"panel": panel, "real": real})
            assert r.status_code == 200, (path, r.text)
    assert list(tmp_path.iterdir()) == []
    assert secret not in caplog.text


# ---- limits and rate limit
def test_oversized_request_is_rejected_before_parsing():
    r = client.post("/api/audit/run", content=b"{" + b" " * (audit_api.MAX_BODY + 10) + b"}", headers={"Content-Type": "application/json"})
    assert r.status_code == 413


def test_too_many_rows_via_api_returns_413():
    r = client.post("/api/audit/preview", json={"panel": "a\n" + "1\n" * 5001, "real": "a\n1\n2\n"})
    assert r.status_code == 413


def test_audit_is_rate_limited():
    body = {"panel": to_csv(HEADERS, likert_rows(40, 1)), "real": to_csv(HEADERS, likert_rows(30, 2))}
    codes = [client.post("/api/audit/preview", json=body).status_code for _ in range(audit_api.RATE_LIMIT[0] + 1)]
    assert codes[:-1] == [200] * audit_api.RATE_LIMIT[0] and codes[-1] == 429


# ---- sample flow end to end
def test_sample_data_flow_end_to_end():
    s = client.get("/api/audit/sample").json()
    body = {"panel": s["panel"], "real": s["real"]}
    pv = client.post("/api/audit/preview", json=body).json()
    assert len(pv["columns"]) == 20 and all(c["on"] for c in pv["columns"]) and pv["n_real"] == 100
    rep = client.post("/api/audit/run", json={**body, "use": [c["name"] for c in pv["columns"]]}).json()
    sm = rep["summary"]
    assert 0.3 < sm["missed"] < 0.65 and sm["n_real"] == 100 and sm["expected_error"]["reliable"] is True
    assert len(rep["map"]["real"]) == 100 and len(rep["examples"]) >= 1 and len(rep["gaps"]) == 5
    assert "misses about" in rep["recommendation"]["headline"]
    html = client.post("/api/audit/report.html", json=body)
    assert html.status_code == 200 and "<svg" in html.text and "<script" not in html.text
    rep_csv = client.post("/api/audit/repaired.csv", json=body)
    rows = list(csv.reader(io.StringIO(rep_csv.text)))
    assert rows[0][-1] == "source" and {r[-1] for r in rows[1:]} == {"AI panel", "statistical augmentation (fitted on your real sample)"}
    real_rows = {tuple(r) for r in list(csv.reader(io.StringIO(s["real"])))[1:]}
    assert not any(tuple(r[:-1]) in real_rows for r in rows[1:] if r[-1].startswith("statistical"))


def test_unticking_columns_changes_the_comparison():
    s = client.get("/api/audit/sample").json()
    names = [c["name"] for c in client.post("/api/audit/preview", json={"panel": s["panel"], "real": s["real"]}).json()["columns"]]
    rep = client.post("/api/audit/run", json={"panel": s["panel"], "real": s["real"], "use": names[:5]}).json()
    assert rep["summary"]["n_questions"] == 5


def test_audit_page_served_and_linked_from_story():
    assert client.get("/audit").status_code == 200
    assert 'href="/audit"' in client.get("/").text
