"""Blindspot demo server: result JSON for the story page plus live join scoring.

Run:  uvicorn blindspot.api.main:app --host 0.0.0.0 --port 8000
Open: http://localhost:8000/  (story)   /join  (phone)   /stage  (projector)
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import time
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..audit.api import router as audit_router
from .adaptive import N_QUESTIONS, AdaptivePlanner
from .followup import REPLY_MAX, FollowUp
from .live import LiveScorer, VisitorFeed
from .room import RoomScorer

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


_hits: dict[str, deque] = defaultdict(deque)
_all_hits: deque = deque()
# A whole room on one venue Wi-Fi shares one public address, so the per-address limit must fit a room;
# the global cap keeps a script from flooding the stage.
RATE_LIMIT = (60, 60.0)          # submissions per address per minute
GLOBAL_LIMIT = (1500, 60.0)      # submissions from everyone per minute


def _room_left(q: deque, limit: tuple[int, float], now: float) -> bool:
    n, window = limit
    while q and now - q[0] > window:
        q.popleft()
    return len(q) < n


def _rate_ok(ip: str) -> bool:
    now = time.time()
    q = _hits[ip]
    if not (_room_left(q, RATE_LIMIT, now) and _room_left(_all_hits, GLOBAL_LIMIT, now)):
        return False
    q.append(now)
    _all_hits.append(now)
    return True


@app.post("/api/live/answer")
def live_answer(body: Answers, request: Request):
    if not _rate_ok(request.client.host if request.client else "unknown"):
        raise HTTPException(429, "Too many tries. Please wait a minute.")
    scorer, mock = _scorer()
    try:
        result = scorer.score(body.answers)
    except ValueError as e:
        raise HTTPException(422, str(e))
    feed.add(result["x"], result["y"], result["covered"])
    return {"mock": mock, **result}


@lru_cache(maxsize=1)
def _room() -> RoomScorer:
    bundle, _ = _load("live_gss")
    return RoomScorer(bundle)


@app.get("/api/room/questions")
def room_questions():
    room = _room()
    return {"version": "room-7", "items": room.questions(), "miss_rate": room.miss_rate, "real_n": len(room.real),
            "crowd_n": room.crowd_n}


@app.post("/api/room/answer")
def room_answer(body: Answers, request: Request):
    """7-question room version, scored against the Claude crowd AND the Flashpoint panel. Nothing stored."""
    if not _rate_ok(request.client.host if request.client else "unknown"):
        raise HTTPException(429, "Too many tries. Please wait a minute.")
    try:
        result = _room().score(body.answers)
    except ValueError as e:
        raise HTTPException(422, str(e))
    if not result["extremes"]:          # all-extreme answer sets are shown their verdict but kept off the stage counts
        feed.add(result["x"], result["y"], result["verdicts"]["claude"]["covered"])
    return result


@lru_cache(maxsize=1)
def _planner() -> AdaptivePlanner:
    bundle, _ = _load("live_gss")
    return AdaptivePlanner(bundle)


@lru_cache(maxsize=1)
def _followup() -> FollowUp:
    return FollowUp()


class PathBody(BaseModel):
    answers: dict[str, int | None] = Field(default_factory=dict)
    path: list[str] = Field(default_factory=list, max_length=N_QUESTIONS)


class ReflectBody(PathBody):
    question: str = Field(max_length=300)
    reply: str = Field(max_length=REPLY_MAX)


def _guard(request: Request) -> None:
    if not _rate_ok(request.client.host if request.client else "unknown"):
        raise HTTPException(429, "Too many tries. Please wait a minute.")


def _verdict(body: PathBody) -> dict:
    try:
        return _planner().score(body.answers, body.path)
    except ValueError as e:                  # PathError and the scorer's "answer at least half" check
        raise HTTPException(422, str(e))


@app.get("/api/adaptive/start")
def adaptive_start():
    return {"version": f"adaptive-{N_QUESTIONS}", "n_questions": N_QUESTIONS, "question": _planner().next({}, []),
            "followup": _followup().available()}


@app.post("/api/adaptive/next")
def adaptive_next(body: PathBody):
    try:
        q = _planner().next(body.answers, body.path)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {"done": q is None, "question": q}


@app.post("/api/adaptive/answer")
def adaptive_answer(body: PathBody, request: Request):
    """Adaptive 7-question version, scored against the Claude crowd and the Flashpoint panel. Nothing stored."""
    _guard(request)
    result = _verdict(body)
    if not result["extremes"]:
        feed.add(result["x"], result["y"], result["verdicts"]["claude"]["covered"])
    return {**result, "followup": _followup().available()}


@app.post("/api/adaptive/followup")
def adaptive_followup(body: PathBody, request: Request):
    _guard(request)
    fu = _followup()
    question = fu.question(_verdict(body))          # None: no key, over today's cap, refused or API error
    if not question:
        raise HTTPException(503, "The AI follow-up is not available right now.")
    return {"question": question, "model": fu.model}


@app.post("/api/adaptive/reflect")
def adaptive_reflect(body: ReflectBody, request: Request):
    _guard(request)
    fu = _followup()
    reflection = fu.reflect(_verdict(body), body.question, body.reply)
    if not reflection:
        raise HTTPException(503, "The AI follow-up is not available right now.")
    return {"reflection": reflection}


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


app.include_router(audit_router)

# /join (the QR code in chapter 8) opens the 7-question room survey; the older 20-question flow stays at /join/full.
for route, page in (("/", "index.html"), ("/join", "you.html"), ("/you", "you.html"), ("/join/full", "join.html"),
                    ("/stage", "stage.html"), ("/audit", "audit.html")):
    app.add_api_route(route, (lambda p=page: FileResponse(WEB / p)), methods=["GET"], include_in_schema=False)

app.mount("/static", StaticFiles(directory=WEB), name="static")
