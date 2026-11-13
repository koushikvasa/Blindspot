"""H4b: persona generation with the run-level budget check (every batch of people is estimated first)."""
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from blindspot.agent.budget import Budget
from blindspot.data.schema import Item, Respondents
from blindspot.generators import claude_personas as cp
from blindspot.generators.claude_client import ResponseCache, Usage


@dataclass
class PersonaGenerator:
    """Standard (non-batch) calls with bounded concurrency: the loop waits on every step, so batches
    would add hours; the price difference is small at this scale and the budget guard still applies."""
    client: object
    cache: ResponseCache
    model: str
    usage: Usage
    budget: Budget
    dataset: str
    items: list[Item]
    seed: int
    label: str = ""
    log: Callable[[str], None] = print

    def people(self, demographics: pd.DataFrame, recipes: list[str], what: str) -> Respondents:
        """One synthetic person per demographics row, each with its own recipe ('' = plain claude_mode)."""
        plan = cp.plan(self.dataset, demographics.reset_index(drop=True), self.items, self.model, recipes=recipes)
        estimate = cp.estimate(plan, self.client, self.cache, batch=False)
        self.budget.check(estimate["cost_usd"], f"{self.label} {what}")
        return cp.execute(plan, self.client, self.cache, self.usage, seed=self.seed, batch=False,
                          log=lambda _m: None).mode
