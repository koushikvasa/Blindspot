"""The one data contract every module uses."""
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Item:
    code: str            # column name in `answers`
    wording: str         # question text shown to people and to Claude
    kind: str            # "ordinal" | "nominal" | "binary"
    options: list[str]   # answer labels in order (ordinal) or as listed
    theme: str           # e.g. "trust", "money", "risk", "personality"
    sensitive: bool = False  # politically charged: allowed in analysis, never in live mode


@dataclass
class Respondents:
    """People × items. Real or synthetic, same shape.

    answers: index = respondent id, columns = item codes, values = option index (int) or NaN
    demographics: index = respondent id, columns like age_band, gender, region, education, income_band
    source: "twin2k" | "gss" | "claude:<recipe_version>" | "baseline:<name>" | ...
    """
    answers: pd.DataFrame
    demographics: pd.DataFrame
    items: list[Item]
    source: str
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        assert list(self.answers.columns) == [i.code for i in self.items], "answers columns must match items order"
        assert self.answers.index.equals(self.demographics.index), "answers and demographics must share ids"
