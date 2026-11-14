"""Match question columns between the AI panel and the real sample by header (fuzzy for minor differences)."""
import difflib
import re
from dataclasses import dataclass

FUZZY_CUTOFF = 0.86
_META = re.compile(
    r"^(#|index|row|id|uuid|email|name|status|version|source|segment|calibration|country|city|device|user agent|"
    r"ip hash|ip|started at|completed at|.*\bid\b.*|.*\brespondent\b.*|.*\btimestamp\b.*|.*\bdate\b.*|.*\bduration\b.*|.*\bquality\b.*|"
    r".*\bflag\b.*|.*\bdefender\b.*|.*\bredem\b.*|.*\bpanel\b.*|.*\bsample source\b.*|.*\bpersona\b.*|.*\bcomment\b.*|.*\bopen ?end\b.*|.*\bweight\b.*)$")


def norm_header(header: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", header.lower()).split())


MAX_METADATA_WORDS = 6          # metadata headers are short ("Respondent ID"); long question wording is never metadata


def is_metadata(header: str) -> bool:
    norm = norm_header(header)
    return len(norm.split()) <= MAX_METADATA_WORDS and bool(_META.match(norm))


@dataclass(frozen=True)
class Match:
    panel: str
    real: str
    method: str          # "exact" | "normalized" | "fuzzy"
    score: float


def match_columns(panel_headers: tuple[str, ...], real_headers: tuple[str, ...]) -> tuple[list[Match], list[str], list[str]]:
    """One-to-one matches in the real sample's column order, plus the unmatched headers of each file."""
    matches: list[Match] = []
    free_panel = list(panel_headers)
    pending = []
    by_norm: dict[str, list[str]] = {}
    for h in free_panel:
        by_norm.setdefault(norm_header(h), []).append(h)
    for r in real_headers:
        if r in free_panel:
            matches.append(Match(free_panel.pop(free_panel.index(r)), r, "exact", 1.0))
        elif (cands := [h for h in by_norm.get(norm_header(r), []) if h in free_panel]):
            matches.append(Match(free_panel.pop(free_panel.index(cands[0])), r, "normalized", 1.0))
        else:
            pending.append(r)
    unmatched_real = []
    for r in pending:
        close = difflib.get_close_matches(norm_header(r), [norm_header(h) for h in free_panel], n=1, cutoff=FUZZY_CUTOFF)
        if close:
            target = next(h for h in free_panel if norm_header(h) == close[0])
            matches.append(Match(free_panel.pop(free_panel.index(target)), r, "fuzzy",
                                 round(difflib.SequenceMatcher(None, norm_header(r), close[0]).ratio(), 3)))
        else:
            unmatched_real.append(r)
    order = {h: i for i, h in enumerate(real_headers)}
    matches.sort(key=lambda m: order[m.real])
    return matches, free_panel, unmatched_real
