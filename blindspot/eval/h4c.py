"""H4c (zero cost): can the agent learn WHOM to ask? GSS test split, never holdout, no model calls.

`python -m blindspot.eval.h4c`
Crowd at each budget = claude_mode with the asked real POOL people swapped in. These crowds INCLUDE REAL
RESPONDENTS (label them so); the near-duplicate guard exempts the swapped-in rows and checks only the
remaining synthetic ones. Budgets 25/50/75/100 (rounds of 25), 10 seeds, 24 GSS items (as H4).

random   : random pool people (equals claude_plus_asked).
targeted : round 1 random. After each round the miss score of every asked person is their Gower distance
           to the nearest claude_mode member; a logistic regression on one-hot demographics (the only
           thing visible before asking) predicts miss-above-the-asked-median; the next round asks 70% of
           the unasked pool people with the highest predicted miss probability and 30% at random.
Also: gaussian_copula fit on the targeted vs random asked people; a self-knowledge estimate of the share
of people claude_mode misses; a fairness audit of claude_mode coverage by demographic group.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from blindspot.agent.guards import near_duplicates
from blindspot.coverage import evaluate, real_distances
from blindspot.data.schema import Respondents
from blindspot.generators.baselines import gaussian_copula
from blindspot.space import encode, gower

from .experiment0 import K, K_VALUES, SEED, _assert_no_holdout, _git_commit, load_inputs
from .experiment2 import swap_in_asked
from .experiment_fp import _agg, claude_mode_full

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
N_SEEDS = 10
ROUNDS = 4
PER_ROUND = 25
TARGETED_SHARE = 0.70
N_BOOT_SEED = 20
N_BOOT_AUDIT = 200
MIN_GROUP = 30
FEATURES = ["age_band", "gender", "education", "income_band", "region", "race"]
AUDIT_COLS = ["age_band", "gender", "education", "income_band", "region"]
STRATEGIES = ("random", "targeted")


def miss_scores(people: Respondents, base: Respondents) -> np.ndarray:
    """Gower distance from each person to the nearest claude_mode member (no overlap counts as 1)."""
    d = np.nan_to_num(gower(encode(people), encode(base), base.items), nan=1.0)
    return d.min(axis=1)


def design_matrix(pool: Respondents) -> pd.DataFrame:
    return pd.get_dummies(pool.demographics[FEATURES].astype(str), dtype=float)


def next_round_targeted(x: pd.DataFrame, asked: list[str], miss: pd.Series, per_round: int, rng) -> tuple[list[str], dict]:
    """Ask 70% highest predicted-miss unasked people + 30% random unasked; returns (ids, coefficients)."""
    unasked = [i for i in x.index if i not in set(asked)]
    label = (miss.loc[asked] > miss.loc[asked].median()).astype(int)
    n_top = int(round(TARGETED_SHARE * per_round))
    if label.nunique() < 2:                                     # nothing to learn from: fall back to random
        return list(rng.choice(unasked, per_round, replace=False)), {}
    model = LogisticRegression(C=1.0, max_iter=1000).fit(x.loc[asked], label)
    p = pd.Series(model.predict_proba(x.loc[unasked])[:, 1], index=unasked)
    top = list(p.sort_values(ascending=False, kind="stable").index[:n_top])
    rest = [i for i in unasked if i not in set(top)]
    rand = list(rng.choice(rest, per_round - n_top, replace=False))
    return top + rand, dict(zip(x.columns, map(float, model.coef_[0])))


def asked_sets(strategy: str, pool: Respondents, x: pd.DataFrame, miss: pd.Series, seed: int
               ) -> tuple[list[list[str]], list[dict]]:
    """Cumulative asked ids after each round, and the logged coefficients that chose each round (round 1: none)."""
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(np.array(pool.answers.index)))
    asked: list[str] = order[:PER_ROUND]
    sets, coefs = [list(asked)], [{}]
    if strategy == "random":
        return [order[:PER_ROUND * (r + 1)] for r in range(ROUNDS)], [{}] * ROUNDS
    for _ in range(1, ROUNDS):
        new, coef = next_round_targeted(x, asked, miss, PER_ROUND, rng)
        asked = asked + new
        sets.append(list(asked))
        coefs.append(coef)
    return sets, coefs


def score(target: Respondents, crowd: Respondents, rr, seed: int) -> dict:
    m = evaluate(target, crowd, target.items, k=K, n_boot=N_BOOT_SEED, seed=seed, k_values=K_VALUES, real_dist=rr)
    return {"coverage": m["coverage"], "precision": m["precision"],
            "k_sensitivity": m["k_sensitivity"], "n_synthetic": m["n_synthetic"]}


def self_knowledge(asked: Respondents, pool: Respondents, base: Respondents) -> float:
    """Estimated share of the population claude_mode misses, from the asked people only.

    Missed = nearest claude_mode member farther than the person's 5th nearest ASKED neighbour (the only
    reference the agent has); asked people are reweighted so gender x age-band cells match the pool.
    The radius from a few asked people is larger than a full-sample radius, so this is biased optimistic."""
    enc = encode(asked)
    dd = np.nan_to_num(gower(enc, enc, asked.items), nan=np.inf)
    np.fill_diagonal(dd, np.inf)
    radius = np.partition(dd, K - 1, axis=1)[:, K - 1]
    missed = (np.nan_to_num(gower(enc, encode(base), base.items), nan=np.inf).min(axis=1) > radius).astype(float)
    cell = lambda r: r.demographics["gender"].astype(str) + "|" + r.demographics["age_band"].astype(str)  # noqa: E731
    share = cell(pool).value_counts(normalize=True)
    counts = cell(asked).value_counts()
    w = cell(asked).map(lambda c: share[c] / counts[c]).to_numpy()
    return float((w * missed).sum() / w.sum())


def pool_truth(pool: Respondents, base: Respondents) -> float:
    """Aux truth: share of POOL people claude_mode misses, with full pool 5-NN radii."""
    rp = real_distances(pool, pool.items)
    radius = np.partition(rp, K - 1, axis=1)[:, K - 1]
    nn = np.nan_to_num(gower(encode(pool), encode(base), base.items), nan=np.inf).min(axis=1)
    return float((nn > radius).mean())


def run_seed(seed: int, ctx: dict) -> dict:
    target, pool, base, x, splits = ctx["target"], ctx["pool"], ctx["base"], ctx["x"], ctx["splits"]
    out = {}
    for strat in STRATEGIES:
        sets, coefs = asked_sets(strat, pool, x, ctx["miss"], seed)
        rows = []
        for r, ids in enumerate(sets):
            asked = pool.subset(ids)
            crowd = swap_in_asked(base, asked)
            synthetic_part = base.subset(list(base.answers.index[len(ids):]))
            rows.append({
                "budget": len(ids),
                "claude_plus_asked": score(target, crowd, ctx["rr"], seed),
                "copula_on_asked": score(target, gaussian_copula(asked, target.demographics, splits, seed),
                                         ctx["rr"], seed),
                "near_dup_synthetic_vs_asked": near_duplicates(synthetic_part, asked).share,
                "self_knowledge_estimate": self_knowledge(asked, pool, base),
                "coefficients": coefs[r]})
        out[strat] = rows
    return out


def fairness(target: Respondents, base: Respondents, rr) -> dict:
    m = evaluate(target, base, target.items, k=K, n_boot=20, seed=SEED, k_values=(K,), real_dist=rr)
    covered = m["_real_covered"]
    rng = np.random.default_rng(SEED)
    rows = {}
    for col in AUDIT_COLS:
        for level, idx in target.demographics.groupby(col).indices.items():
            if len(idx) < MIN_GROUP:
                continue
            c = covered[idx]
            boots = [c[rng.integers(0, len(c), len(c))].mean() for _ in range(N_BOOT_AUDIT)]
            rows[f"{col}={level}"] = {"n": int(len(idx)), "coverage": float(c.mean()),
                                      "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
    return {"overall_coverage": float(covered.mean()), "groups": rows, "min_group_n": MIN_GROUP}


def aggregate(per_seed: list[dict]) -> dict:
    """strategy -> budget -> method -> mean/sd/range of coverage and precision, plus seed-level extras."""
    out = {}
    for strat in STRATEGIES:
        out[strat] = []
        for r in range(ROUNDS):
            rows = [s[strat][r] for s in per_seed]
            entry = {"budget": rows[0]["budget"],
                     "near_dup_synthetic_vs_asked": _agg([x["near_dup_synthetic_vs_asked"] for x in rows]),
                     "self_knowledge_estimate": _agg([x["self_knowledge_estimate"] for x in rows])}
            for method in ("claude_plus_asked", "copula_on_asked"):
                entry[method] = {"coverage": _agg([x[method]["coverage"] for x in rows]),
                                 "precision": _agg([x[method]["precision"] for x in rows]),
                                 "k_sensitivity": {kk: float(np.mean([x[method]["k_sensitivity"][kk]["coverage"]
                                                                      for x in rows])) for kk in ("3", "5", "10")}}
            out[strat].append(entry)
    return out


def paired(per_seed: list[dict]) -> list[dict]:
    """targeted minus random, paired by seed, per budget and method."""
    rows = []
    for r in range(ROUNDS):
        for method in ("claude_plus_asked", "copula_on_asked"):
            d = np.array([s["targeted"][r][method]["coverage"] - s["random"][r][method]["coverage"] for s in per_seed])
            rows.append({"budget": per_seed[0]["random"][r]["budget"], "method": method,
                         "mean_diff": float(d.mean()), "sd": float(d.std(ddof=1)),
                         "seeds_targeted_better": int((d > 0).sum()), "n_seeds": len(d)})
    return rows


def mean_coefficients(per_seed: list[dict]) -> dict:
    """Mean logistic coefficient per feature for each targeted round (rounds 2-4 chose from them)."""
    out = {}
    for r in range(1, ROUNDS):          # coefficients stored at index r are the fit that chose round r + 1
        seq = [s["targeted"][r]["coefficients"] for s in per_seed if s["targeted"][r]["coefficients"]]
        feats = sorted({f for c in seq for f in c})
        mean = {f: float(np.mean([c.get(f, 0.0) for c in seq])) for f in feats}
        out[f"fit_before_round_{r + 1}"] = {"n_seeds": len(seq),
                                            "top": sorted(mean.items(), key=lambda kv: -abs(kv[1]))[:8]}
    return out


def run(out_dir: Path = RESULTS) -> dict:
    started = time.perf_counter()
    gss_in = next(d for d in load_inputs() if d.name == "gss")     # holdout already dropped
    _assert_no_holdout(gss_in.real.answers.index, gss_in.splits, "gss real data")
    full_target, pool = gss_in.real.subset(gss_in.splits["test"]), gss_in.real.subset(gss_in.splits["pool"])
    target = full_target
    base = claude_mode_full(target, [it.code for it in target.items])
    x = design_matrix(pool)
    ctx = {"target": target, "pool": pool, "base": base, "x": x, "splits": gss_in.splits,
           "rr": real_distances(target, target.items), "miss": pd.Series(miss_scores(pool, base), index=pool.answers.index)}
    per_seed = [run_seed(SEED + i, ctx) for i in range(N_SEEDS)]
    base_eval = score(target, base, ctx["rr"], SEED)
    report = {
        "settings": {"budgets": [PER_ROUND * (r + 1) for r in range(ROUNDS)], "per_round": PER_ROUND,
                     "n_seeds": N_SEEDS, "targeted_share": TARGETED_SHARE, "features": FEATURES, "k": K,
                     "n_boot_per_seed": N_BOOT_SEED, "n_items": len(target.items), "n_target": len(target.answers),
                     "n_pool": len(pool.answers), "crowd_label": "includes real respondents (asked pool people)",
                     "git_commit": _git_commit()},
        "claude_mode_budget_0": base_eval,
        "results": aggregate(per_seed), "targeted_minus_random": paired(per_seed),
        "targeted_coefficients": mean_coefficients(per_seed),
        "self_knowledge": {"true_uncovered_test": 1 - base_eval["coverage"],
                           "true_uncovered_pool_fullradius": pool_truth(pool, base)},
        "fairness_claude_mode": fairness(target, base, ctx["rr"]),
        "runtime_seconds": round(time.perf_counter() - started, 1),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "h4c.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (out_dir / "h4c.md").write_text(to_markdown(report), encoding="utf-8")
    return report


# ---------- markdown ----------
def _cell(a: dict) -> str:
    return f"{a['mean']:.3f} ({a['p2_5']:.3f}-{a['p97_5']:.3f})"


def to_markdown(r: dict) -> str:
    s, res = r["settings"], r["results"]
    lines = ["# H4c: learning whom to ask (zero cost, GSS test split)", "",
             f"Crowd = claude_mode with the asked pool people swapped in: **{s['crowd_label']}**. {s['n_items']} items, "
             f"target n = {s['n_target']}, pool n = {s['n_pool']}, {s['n_seeds']} seeds. Cells show the mean "
             f"(2.5-97.5 percentile across seeds). k = {s['k']}. Budget 0 (claude_mode): coverage "
             f"{r['claude_mode_budget_0']['coverage']:.3f}, precision {r['claude_mode_budget_0']['precision']:.3f}. "
             f"Commit `{s['git_commit']}`. Runtime {r['runtime_seconds']} s.", ""]
    for method, title in (("claude_plus_asked", "claude_mode + asked real people"),
                          ("copula_on_asked", "gaussian_copula fit on the asked people")):
        lines += [f"## {title}: coverage / precision", "",
                  "| budget | random coverage | targeted coverage | random precision | targeted precision |",
                  "| --- | --- | --- | --- | --- |"]
        for r_row, t_row in zip(res["random"], res["targeted"]):
            lines.append(f"| {r_row['budget']} | {_cell(r_row[method]['coverage'])} | {_cell(t_row[method]['coverage'])} | "
                         f"{_cell(r_row[method]['precision'])} | {_cell(t_row[method]['precision'])} |")
        lines += [""]
    lines += ["## Targeted minus random (coverage, paired by seed)", "",
              "| budget | method | mean diff | sd | seeds targeted better |", "| --- | --- | --- | --- | --- |"]
    lines += [f"| {d['budget']} | {d['method']} | {d['mean_diff']:+.3f} | {d['sd']:.3f} | "
              f"{d['seeds_targeted_better']}/{d['n_seeds']} |" for d in r["targeted_minus_random"]]
    lines += ["", "## Near-duplicate share (synthetic rows within 0.02 Gower of an asked person; swapped-in rows exempt)", "",
              "| budget | random | targeted |", "| --- | --- | --- |"]
    lines += [f"| {a['budget']} | {a['near_dup_synthetic_vs_asked']['mean']:.3f} | "
              f"{b['near_dup_synthetic_vs_asked']['mean']:.3f} |" for a, b in zip(res["random"], res["targeted"])]
    sk = r["self_knowledge"]
    lines += ["", "## Self-knowledge: estimated share of people claude_mode misses", "",
              f"True uncovered share on the test split: **{sk['true_uncovered_test']:.3f}** (1 - coverage). Auxiliary truth "
              f"on the pool with full-pool radii: {sk['true_uncovered_pool_fullradius']:.3f}. Estimate = asked people "
              "farther from every claude_mode member than from their 5th nearest asked neighbour, reweighted to pool "
              "gender x age-band shares.", "", "| budget | estimate (random asked) | estimate (targeted asked) |",
              "| --- | --- | --- |"]
    lines += [f"| {a['budget']} | {_cell(a['self_knowledge_estimate'])} | {_cell(b['self_knowledge_estimate'])} |"
              for a, b in zip(res["random"], res["targeted"])]
    lines += ["", "## Targeted: mean logistic coefficients (top 8 by size; positive = predicts a miss)", ""]
    for key, v in r["targeted_coefficients"].items():
        lines.append(f"- {key} ({v['n_seeds']} seeds): " + ", ".join(f"{f} {c:+.2f}" for f, c in v["top"]))
    fa = r["fairness_claude_mode"]
    lines += ["", f"## Fairness audit: claude_mode coverage by group (test, overall {fa['overall_coverage']:.3f}, "
              f"groups with n >= {fa['min_group_n']}, bootstrap 95% CI)", "", "| group | n | coverage [CI] |",
              "| --- | --- | --- |"]
    lines += [f"| {g} | {v['n']} | {v['coverage']:.3f} [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}] |"
              for g, v in fa["groups"].items()]
    return "\n".join(lines) + "\n"


def main() -> None:
    run()
    print((RESULTS / "h4c.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
