"""CSV parsing for the audit: the csv module only, in memory only (nothing is ever written to disk).

Limits (CLAUDE.md privacy rules for uploads): 5 MB per file, 5,000 rows, 100 columns.
"""
import csv
import io
from dataclasses import dataclass

MAX_BYTES = 5 * 1024 * 1024
MAX_ROWS = 5000
MAX_COLS = 100
MISSING = {"", "na", "n/a", "nan", "null", "#null!", "-", "."}


class AuditError(ValueError):
    """A problem the user can fix (file, columns, size); `status` is the HTTP code the API returns."""

    def __init__(self, message: str, status: int = 422):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class Table:
    name: str
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    def column(self, j: int) -> list[str]:
        return [r[j] for r in self.rows]


def is_missing(value: str) -> bool:
    return value.strip().lower() in MISSING


def _delimiter(first_line: str) -> str:
    counts = {d: first_line.count(d) for d in (",", ";", "\t")}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def _is_second_header(headers: list[str], row: list[str]) -> bool:
    """Flashpoint-style exports repeat the column names (or the question texts) in a second header row."""
    if row and row[0].strip() == "#":
        return True
    same = sum(1 for h, c in zip(headers, row) if h and h == c)
    return same >= max(2, len([h for h in headers if h]) // 2)


def _dedupe(headers: list[str]) -> tuple[str, ...]:
    seen: dict[str, int] = {}
    out = []
    for j, h in enumerate(headers):
        name = h.strip() or f"column {j + 1}"
        seen[name] = seen.get(name, 0) + 1
        out.append(name if seen[name] == 1 else f"{name} ({seen[name]})")
    return tuple(out)


def parse_csv(text: str, name: str = "file") -> Table:
    if not isinstance(text, str) or not text.strip():
        raise AuditError(f"{name}: the file is empty.")
    if len(text.encode("utf-8", errors="replace")) > MAX_BYTES:
        raise AuditError(f"{name}: larger than 5 MB. Please upload at most 5,000 rows.", 413)
    if "\x00" in text[:8192]:
        raise AuditError(f"{name}: this does not look like a CSV text file. Export your panel as CSV and try again.")
    text = text.lstrip("﻿")
    first_line = text.split("\n", 1)[0]
    reader = csv.reader(io.StringIO(text), delimiter=_delimiter(first_line))
    rows: list[list[str]] = []
    try:
        for r in reader:
            r = [c.strip() for c in r]
            if any(r):
                rows.append(r)
            if len(rows) > MAX_ROWS + 3:
                raise AuditError(f"{name}: more than {MAX_ROWS:,} rows. Please upload at most {MAX_ROWS:,}.", 413)
    except csv.Error as e:
        raise AuditError(f"{name}: could not read it as CSV ({e}).") from None
    if len(rows) < 2:
        raise AuditError(f"{name}: needs a header row and at least one row of answers.")
    width = len(rows[0])
    if width > MAX_COLS:
        raise AuditError(f"{name}: {width} columns; at most {MAX_COLS} are allowed.", 413)
    data = [(r + [""] * width)[:width] for r in rows[1:]]
    if data and _is_second_header(rows[0], data[0]):
        data = data[1:]
    if len(data) > MAX_ROWS:
        raise AuditError(f"{name}: more than {MAX_ROWS:,} rows. Please upload at most {MAX_ROWS:,}.", 413)
    if not data:
        raise AuditError(f"{name}: no answers below the header row.")
    return Table(name, _dedupe(rows[0]), tuple(tuple(r) for r in data))
