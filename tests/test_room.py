from fastapi.testclient import TestClient

from blindspot.api import main
from blindspot.api.room import CODES

client = TestClient(main.app)


def test_room_page_serves():
    for path in ("/you", "/static/js/you.js", "/static/css/you.css"):
        assert client.get(path).status_code == 200, path


def test_room_questions_are_the_seven_short_ones():
    d = client.get("/api/room/questions").json()
    assert [it["code"] for it in d["items"]] == CODES and len(CODES) == 7
    assert set(d["miss_rate"]) == {"claude", "flashpoint"} and d["crowd_n"] == {"claude": 334, "flashpoint": 200}
    assert all(it["question"] and it["wording"] and 3 <= len(it["choices"]) <= 5 for it in d["items"])
    assert all(len(it["question"].split()) <= 7 for it in d["items"])


def test_room_answer_scores_both_crowds_with_aggregates_only():
    items = client.get("/api/room/questions").json()["items"]
    d = client.post("/api/room/answer", json={"answers": {it["code"]: it["choices"][0]["value"] for it in items}}).json()
    assert d["version"] == "room-7" and set(d["verdicts"]) == {"claude", "flashpoint"}
    assert 0 <= d["why"]["real_share"] <= 1 and d["why"]["answered"] == 7
    assert "real" not in d and "crowd" not in d


def test_room_answer_accepts_prefer_not_to_say_but_needs_half():
    items = client.get("/api/room/questions").json()["items"]
    half = {it["code"]: (0 if k < 4 else None) for k, it in enumerate(items)}
    assert client.post("/api/room/answer", json={"answers": half}).status_code == 200
    few = {it["code"]: (0 if k < 3 else None) for k, it in enumerate(items)}
    assert client.post("/api/room/answer", json={"answers": few}).status_code == 422
