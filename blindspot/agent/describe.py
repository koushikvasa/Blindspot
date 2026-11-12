"""Claude calls of the loop (H4): describe a gap as an archetype, and seek the right people to ask next.

Both calls use structured JSON output, go through the response cache (CLAUDE.md rule 7), and are
checked for leaks: only pool people's answers are ever shown. Describing sees ANSWERS only (rule 5:
name patterns by answers, demographics are footnotes); seeking sees demographics of asked people.
"""
import json
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from blindspot.agent.asking import MAX_CONDITIONS, MAX_SEGMENTS, Segment
from blindspot.agent.guards import MAX_ARCHETYPE_WORDS, archetype_problems, truncate_words
from blindspot.data.schema import Item, Respondents
from blindspot.eval.splits import assert_no_leak
from blindspot.generators import claude_personas as cp
from blindspot.generators.claude_client import CallSpec, ResponseCache, Usage, cache_key, call_cached

PROMPT_VERSION = "a1"
WORDING_CHARS = 150
MISSING = {"", "unknown", "nan", "none"}

DESCRIBE_SYSTEM = f"""You help a survey-simulation team. Our synthetic respondents never produce certain kinds of real people. You are shown the answers of a few real respondents who share an unusual answer pattern. Write an ARCHETYPE: a short plain-language description of this pattern that a simulator could use to imitate such a person.

Rules:
- At most {MAX_ARCHETYPE_WORDS} words.
- Describe tendencies and HOW ANSWERS GO TOGETHER (which attitudes and judgments tend to co-occur, what is surprising about the combination). Do not list answers item by item, do not quote answer options, and do not copy any respondent.
- Do not mention demographics. Name the pattern by what these people answer, in 2-5 words.
- Also estimate what share of ALL survey respondents have this pattern: a low and a high estimate in whole percent, using the cluster size and the number of people asked so far.
Respond only with the JSON object."""

SEEK_SYSTEM = """You help a survey team decide which kinds of people to ask next. We have found archetypes (answer patterns) that our simulated respondents miss. For each archetype you see the demographics of the real people who showed it, and the demographics of everyone asked so far.

Propose up to 3 demographic segments to request next, where you expect to find MORE people with these answer patterns. Each segment has 1 or 2 conditions, each condition a field and one of the allowed values for that field, and a one-sentence reason naming the archetype. Use only the allowed fields and values. Prefer segments that are underrepresented so far. Respond only with the JSON object."""

DESCRIBE_SCHEMA = {"type": "object", "additionalProperties": False,
                   "required": ["name", "description", "share_low_pct", "share_high_pct"],
                   "properties": {"name": {"type": "string"}, "description": {"type": "string"},
                                  "share_low_pct": {"type": "integer"}, "share_high_pct": {"type": "integer"}}}
SEEK_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["segments"],
               "properties": {"segments": {"type": "array", "items": {
                   "type": "object", "additionalProperties": False, "required": ["conditions", "reason"],
                   "properties": {"reason": {"type": "string"},
                                  "conditions": {"type": "array", "items": {
                                      "type": "object", "additionalProperties": False,
                                      "required": ["field", "value"],
                                      "properties": {"field": {"type": "string"}, "value": {"type": "string"}}}}}}}}}


@dataclass
class ClaudeCaller:
    """Cached structured-output calls with shared accounting."""
    client: object
    cache: ResponseCache
    model: str
    usage: Usage
    log: Callable[[str], None] = print
    prompts_seen: dict[str, str] = field(default_factory=dict)

    def ask(self, kind: str, system: str, user: str, schema: dict) -> dict | None:
        self.prompts_seen.setdefault(kind, system)
        params = cp.request_params(self.model, [{"type": "text", "text": system}], user, schema)
        key = cache_key(kind=kind, model=self.model, system=system, user=user, prompt=PROMPT_VERSION)
        record = call_cached(self.client, [CallSpec(key, params)], self.cache, self.model, self.usage,
                             use_batch=False, log=lambda _m: None)[key]
        if record is None:
            self.log(f"[{kind}] call failed: {self.usage.errors[-1:]}")
            return None
        try:
            data = json.loads(record["text"])
        except json.JSONDecodeError:
            self.log(f"[{kind}] invalid JSON")
            return None
        return data if isinstance(data, dict) else None


# ---------------- describe ----------------
def cluster_prompt(items: list[Item], cluster: Respondents, n_asked: int) -> str:
    blocks = []
    for n, (rid, row) in enumerate(cluster.answers.iterrows(), start=1):
        lines = [f"- {it.wording[:WORDING_CHARS]} -> {it.options[int(row[it.code])]}"
                 for it in items if pd.notna(row[it.code])]
        blocks.append(f"Respondent {n}:\n" + "\n".join(lines))
    return (f"{len(cluster.answers)} of the {n_asked} real respondents asked so far share this unusual "
            f"pattern.\n\n" + "\n\n".join(blocks) + "\n\nWrite the archetype as JSON.")


@dataclass(frozen=True)
class Description:
    name: str
    text: str
    share_low: float
    share_high: float
    attempts: int
    problems: list[str]
    discarded: bool


def describe_cluster(caller: ClaudeCaller, items: list[Item], cluster: Respondents, n_asked: int,
                     splits: dict[str, list[str]]) -> Description | None:
    """One Claude call (plus one retry if the guard rejects the text). None if it fails or is discarded."""
    assert_no_leak(cluster.answers.index, splits, allowed=("pool",))
    user = cluster_prompt(items, cluster, n_asked)
    problems: list[str] = []
    for attempt in (1, 2):
        prompt = user if attempt == 1 else (
            f"{user}\n\nYour previous description was rejected: {'; '.join(problems)}. Write a different, "
            f"shorter description of the tendencies and how the answers go together, without restating answers.")
        data = caller.ask("describe", DESCRIBE_SYSTEM, prompt, DESCRIBE_SCHEMA)
        if not data or not str(data.get("description", "")).strip():
            problems = ["no usable response"]
            continue
        text = str(data["description"]).strip()
        problems = archetype_problems(text, items)
        low, high = sorted((max(0, min(100, int(data.get("share_low_pct", 0)))),
                            max(0, min(100, int(data.get("share_high_pct", 0))))))
        if not problems:
            return Description(str(data.get("name", "")).strip() or "unnamed pattern", text, low / 100, high / 100,
                               attempt, [], False)
        if all("words" in p for p in problems):          # only too long: truncate instead of retrying again
            return Description(str(data.get("name", "")).strip() or "unnamed pattern", truncate_words(text),
                               low / 100, high / 100, attempt, problems, False)
    caller.log(f"[describe] archetype discarded: {problems}")
    return Description("", "", 0.0, 0.0, 2, problems, True)


# ---------------- seek ----------------
def demographic_counts(demographics: pd.DataFrame, fields: tuple[str, ...]) -> dict[str, dict[str, int]]:
    out = {}
    for f in fields:
        if f in demographics:
            vc = demographics[f].astype(str).value_counts()
            out[f] = {str(k): int(v) for k, v in vc.items() if str(k).lower() not in MISSING}
    return out


def allowed_values(pool_demographics: pd.DataFrame, fields: tuple[str, ...]) -> dict[str, list[str]]:
    return {f: sorted(v for v in pool_demographics[f].astype(str).unique() if v.lower() not in MISSING)
            for f in fields if f in pool_demographics}


def seek_prompt(archetypes: list[dict], asked_demo: pd.DataFrame, allowed: dict[str, list[str]]) -> str:
    fields = tuple(allowed)
    parts = ["Archetypes found so far (members = real people showing the pattern):"]
    for a in archetypes:
        parts.append(f"\n[{a['id']}] {a['name']} ({a['n_members']} members): {a['text']}\n"
                     f"  members' demographics: {json.dumps(a['demographics'])}")
    parts.append(f"\nEveryone asked so far ({len(asked_demo)} people): "
                 f"{json.dumps(demographic_counts(asked_demo, fields))}")
    parts.append(f"\nAllowed fields and values: {json.dumps(allowed)}")
    parts.append("\nPropose up to 3 segments to ask next as JSON.")
    return "\n".join(parts)


def validate_segments(raw: object, allowed: dict[str, list[str]]) -> list[Segment]:
    """Keep only segments using allowed fields/values (canonical casing), at most 3, no duplicates."""
    canon = {f: {v.casefold(): v for v in vals} for f, vals in allowed.items()}
    out, seen = [], set()
    for seg in raw if isinstance(raw, list) else []:
        if not isinstance(seg, dict):
            continue
        conds = []
        for c in seg.get("conditions", []) if isinstance(seg.get("conditions"), list) else []:
            if not isinstance(c, dict):
                continue
            field_, value = str(c.get("field", "")), str(c.get("value", "")).casefold()
            if field_ in canon and value in canon[field_] and all(field_ != f for f, _ in conds):
                conds.append((field_, canon[field_][value]))
        conds = conds[:MAX_CONDITIONS]
        key = tuple(sorted(conds))
        if conds and key not in seen:
            seen.add(key)
            out.append(Segment(tuple(conds), str(seg.get("reason", "")).strip()))
    return out[:MAX_SEGMENTS]


def seek_segments(caller: ClaudeCaller, archetypes: list[dict], asked_demo: pd.DataFrame,
                  pool_demo: pd.DataFrame, asked_ids: list[str], splits: dict[str, list[str]]) -> list[Segment]:
    assert_no_leak(asked_ids, splits, allowed=("pool",))
    allowed = allowed_values(pool_demo, cp.PERSONA_FIELDS)
    data = caller.ask("seek", SEEK_SYSTEM, seek_prompt(archetypes, asked_demo, allowed), SEEK_SCHEMA)
    return validate_segments((data or {}).get("segments"), allowed)
