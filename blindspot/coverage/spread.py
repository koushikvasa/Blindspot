"""Answer-spread summaries shared by the inspect report and coverage evaluation."""
import numpy as np
import pandas as pd

from blindspot.data.schema import Respondents


def top_share(s: pd.Series) -> float:
    vc = s.dropna().value_counts(normalize=True)
    return float(vc.max()) if len(vc) else float("nan")


def norm_entropy(s: pd.Series, k: int) -> float:
    """Shannon entropy of the answer distribution divided by log(k): 0 = everyone agrees, 1 = uniform."""
    p = s.dropna().value_counts(normalize=True).to_numpy()
    if k < 2 or not len(p):
        return float("nan")
    return float(-(p * np.log(p)).sum() / np.log(k))


def item_entropy(resp: Respondents) -> pd.Series:
    return pd.Series({it.code: norm_entropy(resp.answers[it.code], len(it.options)) for it in resp.items})


def family_entropy(resp: Respondents) -> dict[str, float]:
    """Mean normalized entropy per item family."""
    ent = item_entropy(resp)
    fam = pd.Series({it.code: it.family for it in resp.items})
    return {f: float(v) for f, v in ent.groupby(fam, sort=False).mean().items()}
