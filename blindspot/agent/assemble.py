"""H4b step 5: assemble the crowd.

crowd = claude_mode, with some claude_mode people REPLACED by people generated from an accepted recipe.
Each accepted cluster replaces round(share_of_asked * n_target) people, split evenly over its accepted
variants. A replaced slot is a target person whose demographics best match a profile drawn from the
cluster's asked members (most matching fields, ties broken at random). The replaced person keeps their
own demographics (rule 3): only the answers come from the recipe persona.

No real respondent's answers enter the crowd (CLAUDE.md rule 2). The "asked real people swapped in"
variant of the handover is the baseline claude_plus_asked, not part of this method: swapping real rows in
would also fail the near-duplicate guard and make every asked person trivially "covered".
"""
import numpy as np

from blindspot.agent.book import ClusterBook
from blindspot.agent.generate import PersonaGenerator
from blindspot.data.schema import Respondents
from blindspot.generators.claude_personas import PERSONA_FIELDS


def assemble(gen: PersonaGenerator, baseline: Respondents, book: ClusterBook, pool: Respondents, n_asked: int,
             rng: np.random.Generator) -> tuple[Respondents, dict]:
    n = len(baseline.answers)
    fields = [f for f in PERSONA_FIELDS if f in baseline.demographics and f in pool.demographics]
    test_vals = baseline.demographics[fields].astype(str).to_numpy()
    available = np.ones(n, dtype=bool)
    picks: list[int] = []
    recipes: list[str] = []
    per_entry = {}
    for entry in book.accepted():
        want = int(round(len(entry.members) / max(n_asked, 1) * n))
        members = pool.demographics.loc[sorted(entry.members), fields].astype(str).to_numpy()
        placed = 0
        for j in range(want):
            if not available.any():
                break
            score = np.where(available, (test_vals == members[rng.integers(len(members))]).sum(axis=1), -1)
            slot = int(rng.choice(np.flatnonzero(score == score.max())))
            available[slot] = False
            picks.append(slot)
            recipes.append(entry.variants[j % len(entry.variants)].recipe())
            placed += 1
        per_entry[entry.id] = {"wanted": want, "placed": placed, "n_variants": len(entry.variants)}
    info = {"n_replaced": len(picks), "share_replaced": round(len(picks) / n, 4), "per_cluster": per_entry}
    if not picks:
        return baseline, info
    people = gen.people(baseline.demographics.iloc[picks], recipes, "assemble")
    answers = baseline.answers.copy()
    answers.iloc[picks] = people.answers.to_numpy()
    return Respondents(answers, baseline.demographics, baseline.items, "agent_h4b",
                       meta={"replaced_positions": picks}), info
