"""H4b step 2: for one uncovered cluster, Claude writes 3 distinct recipe variants of its answer pattern.

Existing guards apply to every variant: at most 120 words, no item-by-item answer restating, answers
only (no demographics), only pool people shown (assert_no_leak). Variants that fail the copy guard are
dropped; an over-long variant is truncated. A cluster whose call yields no valid variant is a failed attempt.
"""
from dataclasses import dataclass

from blindspot.agent.describe import ClaudeCaller, cluster_prompt
from blindspot.agent.guards import MAX_ARCHETYPE_WORDS, archetype_problems, truncate_words
from blindspot.agent.library import RECIPE_FRAME
from blindspot.data.schema import Item, Respondents
from blindspot.eval.splits import assert_no_leak

N_VARIANTS = 3

PROPOSE_SYSTEM = f"""You help a survey-simulation team. Our synthetic respondents never produce certain kinds of real people. You are shown the answers of a few real respondents who share an unusual answer pattern. Write {N_VARIANTS} DISTINCT recipe variants: different ways of describing this pattern for a simulator (for example emphasising different tendencies or different ways the answers go together), so that at least one of them makes a simulated respondent answer like these people.

Rules for every variant:
- At most {MAX_ARCHETYPE_WORDS} words.
- Describe tendencies and HOW ANSWERS GO TOGETHER. Do not list answers item by item, do not quote answer options, and do not copy any respondent.
- Do not mention demographics. Name each variant by what these people answer, in 2-5 words.
Respond only with the JSON object."""

PROPOSE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["variants"],
                  "properties": {"variants": {"type": "array", "items": {
                      "type": "object", "additionalProperties": False, "required": ["name", "description"],
                      "properties": {"name": {"type": "string"}, "description": {"type": "string"}}}}}}


@dataclass(frozen=True)
class Variant:
    name: str
    text: str

    def recipe(self) -> str:
        return f"{RECIPE_FRAME}{self.name}: {self.text}"


def propose_variants(caller: ClaudeCaller, items: list[Item], cluster: Respondents, n_asked: int,
                     splits: dict[str, list[str]], attempt: int = 1, rejected: list[str] | None = None
                     ) -> tuple[list[Variant], list[dict]]:
    """Up to 3 valid distinct variants plus a log of dropped ones (with the guard problems)."""
    assert_no_leak(cluster.answers.index, splits, allowed=("pool",))
    user = cluster_prompt(items, cluster, n_asked).replace("Write the archetype as JSON.",
                                                           f"Write the {N_VARIANTS} variants as JSON.")
    if attempt > 1:
        user += ("\n\nA previous set of variants did not make simulated respondents answer like these people "
                 f"(rejected: {'; '.join(rejected or [])}). Write {N_VARIANTS} clearly different variants.")
    data = caller.ask(f"propose{attempt}", PROPOSE_SYSTEM, user, PROPOSE_SCHEMA)
    raw = (data or {}).get("variants")
    kept: list[Variant] = []
    dropped: list[dict] = []
    for entry in raw if isinstance(raw, list) else []:
        if not isinstance(entry, dict) or not str(entry.get("description", "")).strip():
            continue
        name, text = str(entry.get("name", "")).strip() or "unnamed pattern", str(entry["description"]).strip()
        problems = archetype_problems(text, items)
        if problems and not all("words" in p for p in problems):
            dropped.append({"name": name, "problems": problems})
            continue
        variant = Variant(name, truncate_words(text))
        if all(variant.text != k.text for k in kept):
            kept.append(variant)
    return kept[:N_VARIANTS], dropped
