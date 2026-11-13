"""Blindspot demo server: result JSON for the story page plus live join scoring.

Run:  uvicorn blindspot.api.main:app --host 0.0.0.0 --port 8000
Open: http://localhost:8000/  (story)   /join  (phone)   /stage  (projector)
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .live import LiveScorer, VisitorFeed

ROOT = Path(__file__).resolve().parents[2]
REAL_DIR = ROOT / "results" / "ui"
MOCK_DIR = ROOT / "web" / "mock"
WEB = ROOT / "web"
NAME_RE = re.compile(r"^[a-z0-9_]{1,64}$")

app = FastAPI(title="Blindspot")
feed = VisitorFeed()


def _load(name: str) -> tuple[dict, bool]:
    if not NAME_RE.match(name):
        raise HTTPException(400, "bad name")
    for folder, mock in ((REAL_DIR, False), (MOCK_DIR, True)):
        f = folder / f"{name}.json"
        if f.exists():
            return json.loads(f.read_text(encoding="utf-8")), mock
    raise HTTPException(404, f"{name} not found")


@app.get("/api/data/{name}")
def data(name: str):
    payload, mock = _load(name)
    return {"mock": mock, "data": payload}


@lru_cache(maxsize=1)
def _scorer() -> tuple[LiveScorer, bool]:
    bundle, mock = _load("live_gss")
    return LiveScorer(bundle), mock


@app.get("/api/live/questions")
def live_questions():
    scorer, mock = _scorer()
    return {"mock": mock, "crowd_label": scorer.crowd_label,
            "items": [{"code": it["code"], "wording": it["wording"], "options": it["options"]} for it in scorer.items]}


class Answers(BaseModel):
    answers: dict[str, int | None]


@app.post("/api/live/answer")
def live_answer(body: Answers):
    scorer, mock = _scorer()
    try:
        result = scorer.score(body.answers)
    except ValueError as e:
        raise HTTPException(422, str(e))
    feed.add(result["x"], result["y"], result["covered"])
    return {"mock": mock, **result}


@app.get("/api/live/recent")
def live_recent(after: int = 0):
    return {"points": feed.since(after)}


@app.post("/api/live/reset")
def live_reset():
    feed.reset()
    return {"ok": True}


@app.get("/api/health")
def health():
    return {"ok": True, "real_files": sorted(p.stem for p in REAL_DIR.glob("*.json")) if REAL_DIR.exists() else []}


for route, page in (("/", "index.html"), ("/join", "join.html"), ("/stage", "stage.html")):
    app.add_api_route(route, (lambda p=page: FileResponse(WEB / p)), methods=["GET"], include_in_schema=False)

app.mount("/static", StaticFiles(directory=WEB), name="static")
