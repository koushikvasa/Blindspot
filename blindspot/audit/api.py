"""HTTP layer of the audit: JSON bodies only (CSV text inside), processed in memory, never written to disk or logged.

Why JSON and not multipart: Starlette spools multipart files above 1 MB to a temp file on disk, which would break the
"nothing is stored" promise, and it would need python-multipart in the image. The page reads the files in the browser
and posts their text.
"""
import csv
import io
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.routing import APIRoute
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

from .labels import short_labels
from .engine import expected_error, inspect_columns, load_calibration, prepare
from .parse import MAX_BYTES, AuditError, parse_csv
from .render import render_html
from .report import build_report, repaired_rows

SAMPLE_DIR = Path(__file__).with_name("sample")
MAX_BODY = 2 * MAX_BYTES + 200_000
RATE_LIMIT = (12, 60.0)                 # at most 12 audit calls per visitor per minute
SHOWN_OPTIONS = 8

_hits: dict[str, deque] = defaultdict(deque)


class AuditBody(BaseModel):
    panel: str = Field(max_length=MAX_BYTES + 1000)
    real: str = Field(max_length=MAX_BYTES + 1000)
    use: list[str] | None = Field(default=None, max_length=200)


def rate_ok(ip: str) -> bool:
    now, (n, window) = time.time(), RATE_LIMIT
    q = _hits[ip]
    while q and now - q[0] > window:
        q.popleft()
    if len(q) >= n:
        return False
    q.append(now)
    return True


class GuardedRoute(APIRoute):
    """Size and rate checks run BEFORE the body is read, so rejected requests cost nothing."""

    def get_route_handler(self):
        original = super().get_route_handler()

        async def handler(request: Request):
            if request.method == "POST":
                if not rate_ok(request.client.host if request.client else "unknown"):
                    raise HTTPException(429, "Too many tries. Please wait a minute.")
                if int(request.headers.get("content-length") or 0) > MAX_BODY:
                    raise HTTPException(413, "The files are too large (5 MB per file, 5,000 rows, 100 columns).")
            return await original(request)

        return handler


router = APIRouter(prefix="/api/audit", route_class=GuardedRoute)


def _tables(body: AuditBody):
    try:
        return parse_csv(body.panel, "AI panel file"), parse_csv(body.real, "Real sample file")
    except AuditError as err:
        raise HTTPException(err.status, str(err)) from None


def option_labels(spec) -> list[str]:
    if spec.kind == "numeric":
        return [f"{spec.lo:g} to {spec.hi:g}"]
    if spec.kind == "ordinal":
        keys = sorted(spec.positions, key=spec.positions.get)
    else:
        keys = list(spec.categories)
    return [spec.labels.get(k, k) for k in keys]


@router.get("/sample")
def sample():
    return {"panel": (SAMPLE_DIR / "ai_panel_flashpoint.csv").read_text(encoding="utf-8"),
            "real": (SAMPLE_DIR / "real_sample_gss_100.csv").read_text(encoding="utf-8"),
            "panel_name": "Flashpoint synthetic panel (sample)", "real_name": "100 random GSS respondents (sample)"}


@router.post("/preview")
def preview(body: AuditBody):
    panel, real = _tables(body)
    try:
        entries, un_panel, un_real = inspect_columns(panel, real)
    except AuditError as err:
        raise HTTPException(err.status, str(err)) from None
    if not entries:
        raise HTTPException(422, "No question columns match between the two files. The column headers must be the same "
                                 "(small differences such as capitals or underscores are fine).")
    cols = []
    lab = short_labels([x["name"] for x in entries])
    for x in entries:
        spec = x["spec"]
        opts = option_labels(spec) if spec else []
        cols.append({"name": x["name"], "label": lab[x["name"]], "panel_header": x["panel_header"], "method": x["method"], "kind": spec.kind if spec else None,
                     "n_options": spec.n_options if spec else 0, "options": opts[:SHOWN_OPTIONS], "more_options": max(0, len(opts) - SHOWN_OPTIONS),
                     "on": spec is not None and x["reason"] is None, "usable": spec is not None, "reason": x["reason"]})
    return {"columns": cols, "unmatched_panel": un_panel, "unmatched_real": un_real, "n_panel": panel.n_rows, "n_real": real.n_rows,
            "expected_error": expected_error(min(real.n_rows, 500), load_calibration())}


def _prepared(body: AuditBody):
    panel, real = _tables(body)
    try:
        return prepare(panel, real, set(body.use) if body.use is not None else None)
    except AuditError as err:
        raise HTTPException(err.status, str(err)) from None


@router.post("/run")
def run(body: AuditBody):
    return build_report(_prepared(body))


@router.post("/report.html")
def report_html(body: AuditBody):
    return HTMLResponse(render_html(build_report(_prepared(body))),
                        headers={"Content-Disposition": 'attachment; filename="blindspot-audit-report.html"'})


@router.post("/repaired.csv")
def repaired_csv(body: AuditBody):
    header, rows = repaired_rows(_prepared(body))
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(header)
    writer.writerows(rows)
    return Response(out.getvalue(), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": 'attachment; filename="blindspot-repaired-crowd.csv"'})
