"""Plain "silicon sampling" with Claude (H3): one synthetic person per target person.

Persona = demographics only (CLAUDE.md rules 3 and 9): a short first-person backstory built from a
fixed list of demographic fields. Never persona text from Twin-2K-500, never answers from any split.

One call per synthetic person answers every item. The response is JSON (structured output; the
configured model rejects forced tool_choice): for each question, `p` = integer percent per option
(normalized after parsing) and `a` = the single option the person would choose.
  claude_mode    : the chosen option `a`
  claude_sampled : one option drawn from `p` with a per-person seeded RNG
Sampling temperature is the API default (1.0); the API rejects an explicit `temperature` field.
"""
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError, field_validator

from blindspot.data.schema import Item, Respondents

from .claude_client import (CallSpec, ResponseCache, Usage, cache_key, call_cached, check_cost,
                            estimate_cost)

ROOT = Path(__file__).resolve().parents[2]
PROMPT_DIR = ROOT / "results/prompts"
PROMPT_VERSION = "p1"
MAX_TOKENS = 4000
# Only these demographic fields may reach a prompt (anything else, e.g. persona text, is ignored).
PERSONA_FIELDS = ("age_band", "gender", "race", "education", "income_band", "region")
MISSING_VALUES = {"", "unknown", "nan", "none"}
OUTPUT_CHARS_PER_TOKEN = 2.5    # digits and punctuation tokenize densely
OUTPUT_ESTIMATE_MARGIN = 1.25

INSTRUCTIONS = """You are helping simulate a survey of adults living in the United States.
The user message describes one survey respondent. Answer every question below as that person would.

Return {"answers": [...]} with one entry per question, in order, each with:
- "q": the question number (1 for q1, 2 for q2, ...);
- "p": your estimate, in whole percent, of how likely this person is to choose each option, listed in the option order shown (one number per option, summing to 100);
- "a": the option number this person would actually choose.

Answer as the person would, not as you would. Respond only with the JSON object."""


def recipe_version(recipe: str) -> str:
    return "empty" if not recipe.strip() else "r" + hashlib.sha256(recipe.encode()).hexdigest()[:10]


def persona_text(row: pd.Series) -> str:
    """First-person backstory from the allowed demographic fields only."""
    def get(name: str) -> str | None:
        value = row.get(name)
        text = "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value).strip()
        return None if text.lower() in MISSING_VALUES else text

    parts = []
    age, gender = get("age_band"), get("gender")
    if age or gender:
        who = " ".join(p for p in [gender.lower() if gender else "person"] if p)
        parts.append(f"I am a {who}" + (f" in the {age} age group." if age else "."))
    if race := get("race"):
        parts.append(f"My race or ethnicity is {race}.")
    if edu := get("education"):
        parts.append(f"The highest level of education I have completed is: {edu}.")
    if income := get("income_band"):
        parts.append(f"My total family income last year was {income}.")
    if region := get("region"):
        parts.append(f"I live in the {region} region of the United States.")
    return " ".join(parts) or "I am an adult living in the United States."


def question_block(items: list[Item]) -> str:
    lines = ["Questions:"]
    for n, it in enumerate(items, start=1):
        options = "; ".join(f"{i} = {label}" for i, label in enumerate(it.options))
        lines.append(f"q{n}. {it.wording}\n    Options: {options}")
    return "\n".join(lines)


def build_system(items: list[Item], recipe: str = "") -> list[dict]:
    """Stable, cacheable instructions + questions; the recipe (H4) is appended after them."""
    blocks = [{"type": "text", "text": f"{INSTRUCTIONS}\n\n{question_block(items)}",
               "cache_control": {"type": "ephemeral"}}]
    if recipe.strip():
        blocks.append({"type": "text", "text": recipe.strip()})
    return blocks


def build_user(persona: str) -> str:
    return f"Respondent description:\n{persona}\n\nAnswer all questions as this person, in JSON."


def response_schema(items: list[Item]) -> dict:
    """One array of {q, p, a}: a single small sub-schema (one property per question compiles to a
    grammar the API rejects as too large)."""
    one = {"type": "object", "additionalProperties": False, "required": ["q", "p", "a"],
           "properties": {"q": {"type": "integer"}, "p": {"type": "array", "items": {"type": "integer"}},
                          "a": {"type": "integer"}}}
    return {"type": "object", "additionalProperties": False, "required": ["answers"],
            "properties": {"answers": {"type": "array", "items": one}}}


def thinking_off(model: str) -> dict:
    """Request fields that turn thinking off (answers only); the syntax differs by model."""
    if model.startswith("claude-sonnet-5-5"):
        return {"thinking": {"type": "between_tools"}}      # 'disabled' is a 400 on Sonnet 5.5
    if model.startswith("claude-opus-5-5"):
        return {"output_config": {"effort": "low"}}          # thinking cannot be disabled; lowest effort
    return {"thinking": {"type": "disabled"}}               # Haiku 5.5 / older: allowed at effort <= high


def request_params(model: str, system: list[dict], user: str, schema: dict) -> dict:
    params = {"model": model, "max_tokens": MAX_TOKENS, "system": system,
              "messages": [{"role": "user", "content": user}]}
    extra = thinking_off(model)
    output_config = {**extra.pop("output_config", {}), "format": {"type": "json_schema", "schema": schema}}
    return {**params, **extra, "output_config": output_config}


# ---------- parsing ----------
class ItemAnswer(BaseModel):
    p: list[float]
    a: int

    @field_validator("p")
    @classmethod
    def _non_negative(cls, v: list[float]) -> list[float]:
        if any(x < 0 for x in v):
            raise ValueError("negative probability")
        if sum(v) <= 0:
            raise ValueError("probabilities sum to zero")
        return v


@dataclass
class Parsed:
    probs: dict[str, np.ndarray] = field(default_factory=dict)
    answers: dict[str, int] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    def complete(self, items: list[Item]) -> bool:
        return all(it.code in self.probs for it in items)


def parse_record(record: dict | None, items: list[Item]) -> Parsed:
    out = Parsed()
    if record is None:
        out.problems.append("no response")
        return out
    if record.get("stop_reason") not in ("end_turn", None):
        out.problems.append(f"stop_reason={record.get('stop_reason')}")
    try:
        data = json.loads(record.get("text") or "")
    except json.JSONDecodeError:
        out.problems.append("invalid JSON")
        return out
    entries = data.get("answers") if isinstance(data, dict) else None
    by_q: dict[int, dict] = {}
    for entry in entries if isinstance(entries, list) else []:
        if isinstance(entry, dict) and isinstance(entry.get("q"), int):
            by_q.setdefault(entry["q"], entry)          # first answer per question wins
    for n, it in enumerate(items, start=1):
        raw = by_q.get(n)
        if raw is None:
            out.problems.append(f"{it.code}: missing")
            continue
        try:
            ans = ItemAnswer.model_validate(raw)
        except ValidationError as e:
            out.problems.append(f"{it.code}: {e.errors()[0]['msg']}")
            continue
        if len(ans.p) != len(it.options) or not 0 <= ans.a < len(it.options):
            out.problems.append(f"{it.code}: wrong option count or answer out of range")
            continue
        p = np.asarray(ans.p, dtype=float)
        out.probs[it.code] = p / p.sum()
        out.answers[it.code] = ans.a
    return out


def merge(first: Parsed, retry: Parsed) -> Parsed:
    """Prefer the retry's valid items, fall back to the first attempt's; the rest stay missing."""
    merged = Parsed(probs={**first.probs, **retry.probs}, answers={**first.answers, **retry.answers},
                    problems=first.problems + [f"retry: {p}" for p in retry.problems])
    return merged


def sample_answers(probs: dict[str, np.ndarray], items: list[Item], seed: int, person: int) -> dict[str, float]:
    """One option per item drawn from `probs`, deterministic for (seed, person)."""
    rng = np.random.default_rng([seed, person])
    return {it.code: float(rng.choice(len(probs[it.code]), p=probs[it.code])) if it.code in probs else np.nan
            for it in items}


# ---------- planning, estimating, executing ----------
@dataclass
class Plan:
    dataset: str
    model: str
    items: list[Item]
    recipe: str
    demographics: pd.DataFrame
    personas: list[str]
    specs: list[CallSpec]
    recipes: list[str] | None = None   # per-person recipes (H4); None = `recipe` for everyone

    def retry_spec(self, i: int) -> CallSpec:
        base = self.specs[i]
        return CallSpec(cache_key(base=base.key, attempt=1), base.params)


def plan(dataset: str, demographics: pd.DataFrame, items: list[Item], model: str, recipe: str = "",
         recipes: list[str] | None = None) -> Plan:
    """One call spec per target person. `recipes` (H4) gives each person their own recipe text;
    otherwise everyone gets `recipe`. Identical (persona, recipe) pairs get independent replicates."""
    if recipes is not None and len(recipes) != len(demographics):
        raise ValueError(f"need {len(demographics)} recipes, got {len(recipes)}")
    per_person = recipes if recipes is not None else [recipe] * len(demographics)
    schema = response_schema(items)
    systems = {r: build_system(items, r) for r in set(per_person)}
    personas = [persona_text(row) for _, row in demographics.iterrows()]
    seen: dict[tuple[str, str], int] = {}
    specs = []
    for persona, rec in zip(personas, per_person):
        replicate = seen.get((persona, rec), 0)
        seen[(persona, rec)] = replicate + 1
        user = build_user(persona)
        system = systems[rec]
        key = cache_key(model=model, system=system, user=user, items=[i.code for i in items],
                        recipe=recipe_version(rec), prompt=PROMPT_VERSION, replicate=replicate, attempt=0)
        specs.append(CallSpec(key, request_params(model, system, user, schema)))
    return Plan(dataset, model, items, recipe, demographics, personas, specs, recipes)


def _output_tokens_estimate(items: list[Item]) -> int:
    example = {f"q{n}": {"p": [round(100 / len(it.options))] * len(it.options), "a": 0}
               for n, it in enumerate(items, start=1)}
    return int(len(json.dumps(example)) / OUTPUT_CHARS_PER_TOKEN * OUTPUT_ESTIMATE_MARGIN)


def estimate(p: Plan, client, cache: ResponseCache, batch: bool = True) -> dict:
    """Cost of the uncached calls; input tokens from the free count_tokens endpoint."""
    todo = [s for s in p.specs if cache.get(s.key) is None]
    if not todo:
        return estimate_cost(0, 0, 0, p.model, batch)
    params = todo[0].params
    counted = client.messages.count_tokens(model=p.model, system=params["system"], messages=params["messages"])
    schema_tokens = len(json.dumps(params["output_config"])) // 3   # structured-output overhead, rough
    return estimate_cost(len(todo), counted.input_tokens + schema_tokens, _output_tokens_estimate(p.items),
                         p.model, batch)


def save_prompts(p: Plan, prompt_dir: Path = PROMPT_DIR) -> None:
    prompt_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{p.dataset}_{recipe_version(p.recipe)}_{PROMPT_VERSION}"
    params = p.specs[0].params
    (prompt_dir / f"{stem}_system.txt").write_text(
        "\n\n---\n\n".join(b["text"] for b in params["system"]), encoding="utf-8")
    (prompt_dir / f"{stem}_request_settings.json").write_text(json.dumps(
        {k: v for k, v in params.items() if k not in ("system", "messages")}, indent=1), encoding="utf-8")
    (prompt_dir / f"{stem}_user_prompts.txt").write_text(
        "\n\n".join(sorted({build_user(x) for x in p.personas})), encoding="utf-8")


@dataclass
class ClaudeCrowds:
    mode: Respondents
    sampled: Respondents
    failures: dict[str, list[str]]
    usage: Usage


def _crowd(rows: list[dict[str, float]], p: Plan, name: str) -> Respondents:
    ids = pd.Index([f"{name}_{i:05d}" for i in range(len(rows))])
    answers = pd.DataFrame(rows, index=ids, columns=[i.code for i in p.items], dtype="float64")
    demo = p.demographics.reset_index(drop=True).set_axis(ids)
    return Respondents(answers, demo, p.items, f"claude:{name}:{recipe_version(p.recipe)}",
                       meta={"model": p.model, "prompt_version": PROMPT_VERSION,
                             "target_ids": [str(i) for i in p.demographics.index]})


def execute(p: Plan, client, cache: ResponseCache, usage: Usage, seed: int = 42, batch: bool = True,
            cost_limit: float | None = None, log=print) -> ClaudeCrowds:
    records = call_cached(client, p.specs, cache, p.model, usage, batch, log)
    parsed = [parse_record(records[s.key], p.items) for s in p.specs]
    retry_idx = [i for i, x in enumerate(parsed) if not x.complete(p.items)]
    if retry_idx:
        retry_specs = [p.retry_spec(i) for i in retry_idx]
        if cost_limit is not None:
            per_call = _output_tokens_estimate(p.items)
            check_cost(estimate_cost(len(retry_specs), 4 * per_call, per_call, p.model, batch),
                       max(cost_limit - usage.cost_usd, 0.0))
        log(f"[retry] {len(retry_specs)} responses incomplete; retrying once")
        retried = call_cached(client, retry_specs, cache, p.model, usage, batch, log)
        for i, spec in zip(retry_idx, retry_specs):
            parsed[i] = merge(parsed[i], parse_record(retried[spec.key], p.items))
    failures = {f"person_{i:05d}": x.problems for i, x in enumerate(parsed) if not x.complete(p.items)}
    mode_rows = [{it.code: float(x.answers[it.code]) if it.code in x.answers else np.nan for it in p.items}
                 for x in parsed]
    sampled_rows = [sample_answers(x.probs, p.items, seed, i) for i, x in enumerate(parsed)]
    return ClaudeCrowds(_crowd(mode_rows, p, "claude_mode"), _crowd(sampled_rows, p, "claude_sampled"),
                        failures, usage)
