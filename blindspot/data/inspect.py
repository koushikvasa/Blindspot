"""Sanity report for H1: `python -m blindspot.data.inspect`.

Per dataset and file: n people, n items, missingness, top-option share and normalized entropy per
item, per-family answer spread, and for Twin the per-item agreement of gpt41mini_twin and retest
with real. Also creates (once) and reports the saved pool/test/holdout splits.
No coverage metrics here (that is H2).
"""
import pandas as pd

from blindspot.coverage.spread import norm_entropy, top_share
from blindspot.eval.splits import SPLIT_NAMES, get_or_make_splits

from . import gss, twin2k
from .schema import Respondents

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 200)


def item_table(r: Respondents) -> pd.DataFrame:
    rows = []
    for it in r.items:
        s = r.answers[it.code]
        rows.append({"item": it.code, "family": it.family, "k": len(it.options), "missing": s.isna().mean(),
                     "top": top_share(s), "entropy": norm_entropy(s, len(it.options)),
                     "sensitive": it.sensitive, "verified": it.verified})
    return pd.DataFrame(rows).set_index("item")


def agreement(a: pd.DataFrame, b: pd.DataFrame) -> pd.Series:
    """Share of people giving the identical option on both, among those answering both."""
    both = a.notna() & b.notna()
    return (a.eq(b) & both).sum() / both.sum()


def print_header(title: str) -> None:
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")


def describe(name: str, r: Respondents) -> pd.DataFrame:
    t = item_table(r)
    print(f"\n--- {name}: n people = {len(r.answers)}, n items = {len(r.items)}, "
          f"live-mode items = {sum(i.live_ok for i in r.items)}")
    print(f"missing: max {t.missing.max():.1%}; top-option share range {t.top.min():.0%}-{t.top.max():.0%}; "
          f"entropy range {t.entropy.min():.2f}-{t.entropy.max():.2f}")
    return t


def report_twin() -> None:
    print_header("Twin-2K-500")
    data = twin2k.load()
    tables = {name: describe(name, r) for name, r in data.items()}
    real = data["real"].answers
    wide = pd.DataFrame({
        "family": tables["real"].family,
        "miss_real": tables["real"].missing,
        "top_real": tables["real"].top, "top_twin": tables["gpt41mini_twin"].top,
        "top_retest": tables["retest"].top,
        "H_real": tables["real"].entropy, "H_twin": tables["gpt41mini_twin"].entropy,
        "H_retest": tables["retest"].entropy,
        "agree_twin": agreement(real, data["gpt41mini_twin"].answers),
        "agree_retest": agreement(real, data["retest"].answers),
    })
    print("\nPer item (top = top-option share, H = normalized entropy, agree = same option as real):")
    print(wide.round(2).to_string())
    print("\nPer family (means over items):")
    print(wide.groupby("family", sort=False).mean(numeric_only=True).drop(columns="miss_real").round(2).to_string())
    report_splits("twin2k", real.index)


def report_gss() -> None:
    print_header("GSS 2024 (main cross-section sample==13, ballot 1)")
    r = gss.load()
    print(f"file rows = {r.meta['n_file_rows']}; target set = {len(r.answers)}; "
          f"weight non-missing = {r.demographics['weight'].notna().mean():.0%}")
    t = describe("real", r)
    print(t.round(2).to_string())
    print("\nPer family (means over items):")
    print(t.groupby("family", sort=False)[["missing", "top", "entropy"]].mean().round(2).to_string())
    report_splits("gss", r.answers.index)


def report_splits(dataset: str, ids: pd.Index) -> None:
    splits = get_or_make_splits(dataset, ids)
    sizes = ", ".join(f"{n}={len(splits[n])}" for n in SPLIT_NAMES)
    print(f"\nsplits_{dataset}.json: {sizes} (seed 42, 50/30/20)")


def main() -> None:
    report_twin()
    report_gss()


if __name__ == "__main__":
    main()
