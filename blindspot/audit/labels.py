"""Short readable labels for question headers (a real survey export uses the full question wording as the header)."""
import re

LIMIT = 45
_SENTENCE = re.compile(r"(?<=[.?!])\s+")
_INSTRUCTION = re.compile(r"\b(scale|means?)\b", re.I)


def short_label(header: str, limit: int = LIMIT) -> str:
    """The distinguishing part of a long header: text after the last colon, else the last sentence that is not a
    scale instruction, cut at a word boundary. Short headers (codes, one or two words) are returned unchanged."""
    text = " ".join(header.split())
    if len(text) <= limit:
        return text.rstrip(" .?!:") or text
    tail = text.rsplit(":", 1)[1].strip() if ":" in text else ""
    if len(tail) >= 8 and tail.count(",") < 2 and " or " not in tail:      # a stem after the colon, not a list of options
        text = tail
    else:
        sentences = [s for s in _SENTENCE.split(text) if s]
        keep = [s for s in sentences if not _INSTRUCTION.search(s)] or sentences
        text = keep[-1] if keep else text
    text = text.strip().rstrip(" .?!:")
    if len(text) <= limit:
        return text
    cut = text[: limit - 1].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return (cut or text[: limit - 1]) + "…"


def short_labels(headers: list[str]) -> dict[str, str]:
    """header -> unique short label (duplicates get a number)."""
    out: dict[str, str] = {}
    seen: dict[str, int] = {}
    for h in headers:
        label = short_label(h)
        seen[label] = seen.get(label, 0) + 1
        out[h] = label if seen[label] == 1 else f"{label} ({seen[label]})"
    return out
