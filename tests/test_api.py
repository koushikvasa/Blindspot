import json
from pathlib import Path

from fastapi.testclient import TestClient

from blindspot.api import main
from blindspot.api.live import LiveScorer

client = TestClient(main.app)


def test_pages_and_static_serve():
    for path in ("/", "/join", "/stage", "/static/js/map.js", "/static/css/app.css", "/static/vendor/qrcode.js"):
        assert client.get(path).status_code == 200, path


def test_data_falls_back_to_mock_and_rejects_bad_names():
    r = client.get("/api/data/summary")
    assert r.status_code == 200 and "datasets" in r.json()["data"]
    assert client.get("/api/data/..%2Fsecrets").status_code in (400, 404)
    assert client.get("/api/data/Not-Valid").status_code == 400
    assert client.get("/api/data/does_not_exist").status_code == 404


def test_ui_files_contain_no_ids_or_demographic_reasons():
    for f in Path(main.MOCK_DIR).glob("*.json"):
        text = f.read_text()
        for banned in ("TWIN_ID", "respondent_id", "\"reason\"", "demographic"):
            assert banned not in text, (f.name, banned)


def test_live_questions_hide_internal_fields():
    d = client.get("/api/live/questions").json()
    assert d["items"] and set(d["items"][0]) == {"code", "wording", "options"}


def test_live_answer_scores_and_feeds_stage(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # any accidental relative write would land here
    items = client.get("/api/live/questions").json()["items"]
    answers = {it["code"]: 0 for it in items}
    before = client.get("/api/live/recent").json()["points"]
    r = client.post("/api/live/answer", json={"answers": answers})
    assert r.status_code == 200
    d = r.json()
    assert isinstance(d["covered"], bool) and 0 <= d["x"] <= 1 and 0 <= d["y"] <= 1
    after = client.get("/api/live/recent").json()["points"]
    assert len(after) == len(before) + 1
    assert set(after[-1]) == {"seq", "x", "y", "covered", "t"}  # no answers kept
    assert list(tmp_path.iterdir()) == []  # nothing written


def test_live_answer_validation():
    items = client.get("/api/live/questions").json()["items"]
    assert client.post("/api/live/answer", json={"answers": {items[0]["code"]: 99}}).status_code == 422
    assert client.post("/api/live/answer", json={"answers": {}}).status_code == 422


def test_scorer_identical_person_is_covered():
    b = {"items": [{"code": "a", "wording": "", "options": ["x", "y"], "family": "f"},
                   {"code": "b", "wording": "", "options": ["x", "y", "z"], "family": "g"}],
         "real": [[0, 0], [0, 0.5], [1, 1], [1, 0.5], [0, 1], [1, 0]],
         "real_xy": [[0.1, 0.1], [0.2, 0.2], [0.9, 0.9], [0.8, 0.8], [0.1, 0.9], [0.9, 0.1]],
         "crowd": [[0, 0]], "base_rate_uncovered": 0.5}
    s = LiveScorer(b)
    assert s.score({"a": 0, "b": 0})["covered"] is True
    assert s.score({"a": 1, "b": 2})["covered"] is False
