"""H4b step 3: keep a recipe only if it makes synthetic people more like the cluster's asked members.

For each recipe: generate N people whose demographics are drawn (with replacement) from the cluster's
asked members, compute each person's nearest Gower distance to the cluster's asked members, and compare
the median with that of plain claude_mode people with the SAME demographics (the control). Accept only
if the recipe's median is strictly lower. Only pool people are used. The ablation accepts everything.
"""
import numpy as np

from blindspot.agent.generate import PersonaGenerator
from blindspot.agent.propose import Variant
from blindspot.data.schema import Item, Respondents
from blindspot.space import encode, gower

N_VERIFY = 10


def nearest_to_members(people: Respondents, members: Respondents, items: list[Item]) -> np.ndarray:
    d = np.nan_to_num(gower(encode(people), encode(members), items), nan=np.inf)
    return d.min(axis=1)


def _median(values: np.ndarray) -> float:
    return float(np.median(values))


def verify_variants(gen: PersonaGenerator, variants: list[Variant], members: Respondents, items: list[Item],
                    rng: np.random.Generator, n_people: int = N_VERIFY, accept_all: bool = False) -> list[dict]:
    """One row per variant: medians, accept decision, numbers. `accept_all` skips generation (ablation)."""
    if accept_all or not variants:
        return [{"name": v.name, "accepted": True, "median_recipe": None, "median_control": None,
                 "n_people": 0, "verified": False} for v in variants]
    demo = members.demographics.iloc[rng.integers(0, len(members.answers), n_people)]
    control = _median(nearest_to_members(gen.people(demo, [""] * n_people, "verify control"), members, items))
    rows = []
    for v in variants:
        med = _median(nearest_to_members(gen.people(demo, [v.recipe()] * n_people, f"verify {v.name}"),
                                         members, items))
        rows.append({"name": v.name, "accepted": bool(med < control), "median_recipe": round(med, 5),
                     "median_control": round(control, 5), "n_people": n_people, "verified": True})
    return rows
