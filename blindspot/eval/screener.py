"""Screener targeting simulation (zero cost): can 3-4 screening questions find the people the AI crowd misses?

`python -m blindspot.eval.screener`
GSS, 20 live items. Offline simulation, so the true gap is known: a pool person is "missed" when no
claude_mode crowd member is inside their ball (radius = distance to their 5th nearest OTHER pool person).
Per seed the pool is split in random halves: P1 trains the screener (greedy forward selection of items,
logistic regression on each item's value and its squared distance from the scale midpoint, 5-fold CV AUC),
P2 is the recruiting pool. The acceptance threshold is the 60th percentile of P1 scores (~40% accepted).
random   : first B people of P2 in a seeded random order get the full survey.
screened : the same order; only people the screener flags get the full survey, until B are reached.
Repair: gaussian_copula fit on the recruits, and claude_mode with the recruits swapped in (INCLUDES REAL
RESPONDENTS). Scored on the TEST split. Holdout is never touched. 10 seeds.
"""
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from blindspot.agent.guards import near_duplicates
from blindspot.coverage import evaluate, real_distances
from blindspot.generators.baselines import gaussian_copula
from blindspot.space import encode, gower

from .experiment0 import K, K_VALUES, N_BOOT, SEED, _assert_no_holdout, _git_commit, load_inputs
from .experiment2 import swap_in_asked
from .experiment_fp import _agg, claude_mode_full, restrict

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
BUDGETS = (25, 50, 75, 100)
N_SEEDS = 10
SCREEN_SIZES = (3, 4)
ACCEPT_TARGET = 0.40
N_FOLDS = 5
RECRUITS = ("random", "screened_3", "screened_4")
METHODS = ("copula_on_recruits", "claude_plus_recruits")


# ---------- the true gap ----------
def missed_labels(pool, base) -> np.ndarray:
    """True where no claude_mode member is within the person's pool 5-NN radius (offline truth)."""
    rp = real_distances(pool, pool.items)
    radius = np.partition(rp, K - 1, axis=1)[:, K - 1]
    nn = np.nan_to_num(gower(encode(pool), encode(base), base.items), nan=np.inf).min(axis=1)
    return nn > radius


# ---------- screener ----------
def design(x: np.ndarray, cols: list[int]) -> np.ndarray:
    """Per item: value and squared distance from the scale midpoint (NaN -> midpoint)."""
    v = np.nan_to_num(x[:, cols], nan=0.5)
    return np.hstack([v, (v - 0.5) ** 2])


def cv_auc(x: np.ndarray, y: np.ndarray, cols: list[int], seed: int) -> float:
    aucs = []
    for tr, te in StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(x, y):
        model = LogisticRegression(C=1.0, max_iter=1000).fit(design(x[tr], cols), y[tr])
        aucs.append(roc_auc_score(y[te], model.predict_proba(design(x[te], cols))[:, 1]))
    return float(np.mean(aucs))


def forward_select(x: np.ndarray, y: np.ndarray, n_items: int, seed: int) -> list[int]:
    """Greedy forward selection by cross-validated AUC on the training half only."""
    chosen: list[int] = []
    for _ in range(n_items):
        best = max((j for j in range(x.shape[1]) if j not in chosen), key=lambda j: cv_auc(x, y, chosen + [j], seed))
        chosen.append(best)
    return chosen


class Screener:
    def __init__(self, x_train: np.ndarray, y_train: np.ndarray, cols: list[int]):
        self.cols = cols
        self.model = LogisticRegression(C=1.0, max_iter=1000).fit(design(x_train, cols), y_train)
        self.threshold = float(np.quantile(self.score(x_train), 1 - ACCEPT_TARGET))

    def score(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(design(x, self.cols))[:, 1]

    def flags(self, x: np.ndarray) -> np.ndarray:
        return self.score(x) >= self.threshold


def screener_report(s: Screener, x: np.ndarray, y: np.ndarray) -> dict:
    flag, sc = s.flags(x), s.score(x)
    return {"auc": float(roc_auc_score(y, sc)), "accept_rate": float(flag.mean()), "base_rate_missed": float(y.mean()),
            "precision": float(y[flag].mean()) if flag.any() else float("nan"),
            "recall": float(flag[y].mean()) if y.any() else float("nan")}


# ---------- recruitment ----------
def recruit(order: list[str], flags: np.ndarray | None, budget: int) -> tuple[list[str], int]:
    """People who get the full survey (random: first `budget`; screened: first `budget` flagged) and how many were screened."""
    if flags is None:
        return order[:budget], budget
    taken, screened = [], 0
    for pid, flag in zip(order, flags):
        screened += 1
        if flag:
            taken.append(pid)
            if len(taken) == budget:
                break
    return taken, screened


def score(target, crowd, rr, seed: int, synthetic, recruits, n_boot: int) -> dict:
    m = evaluate(target, crowd, target.items, k=K, n_boot=n_boot, seed=seed, k_values=K_VALUES, real_dist=rr)
    return {"coverage": m["coverage"], "precision": m["precision"], "coverage_ci": m["coverage_ci"],
            "precision_ci": m["precision_ci"], "k_sensitivity": m["k_sensitivity"],
            "near_dup_vs_recruits": near_duplicates(synthetic, recruits).share}


def run_seed(seed: int, ctx: dict) -> dict:
    pool, target, base, y_all = ctx["pool"], ctx["target"], ctx["base"], ctx["missed"]
    ids = np.array(pool.answers.index)
    perm = np.random.default_rng(seed).permutation(len(ids))
    half = len(ids) // 2
    p1, p2 = ids[perm[:half]], ids[perm[half:]]
    x_all = encode(pool)
    pos = {i: n for n, i in enumerate(ids)}
    x1, y1 = x_all[[pos[i] for i in p1]], y_all[[pos[i] for i in p1]]
    x2, y2 = x_all[[pos[i] for i in p2]], y_all[[pos[i] for i in p2]]
    screeners = {}
    for n_items in SCREEN_SIZES:
        cols = forward_select(x1, y1, n_items, seed)
        screeners[f"screened_{n_items}"] = Screener(x1, y1, cols)
    order = list(p2)                                          # P2 is already in seeded random order
    out = {"screener": {}, "recruit": {}}
    for name, s in screeners.items():
        out["screener"][name] = {**screener_report(s, x2, y2), "items": [pool.items[c].code for c in s.cols]}
    flags = {name: s.flags(x2) for name, s in screeners.items()}
    for b in BUDGETS:
        out["recruit"][b] = {}
        for rname in RECRUITS:
            taken, n_screened = recruit(order, None if rname == "random" else flags[rname], b)
            recruits = pool.subset(taken)
            cop = gaussian_copula(recruits, target.demographics, ctx["splits"], seed)
            plus = swap_in_asked(base, recruits)
            synthetic = base.subset(list(base.answers.index[len(taken):]))
            out["recruit"][b][rname] = {
                "n_recruited": len(taken), "n_screened": n_screened,
                "share_missed_among_recruits": float(y_all[[pos[i] for i in taken]].mean()),
                "copula_on_recruits": score(target, cop, ctx["rr"], seed, cop, recruits, N_BOOT),
                "claude_plus_recruits": score(target, plus, ctx["rr"], seed, synthetic, recruits, N_BOOT)}
    return out


# ---------- aggregation ----------
def _metric(rows: list[dict]) -> dict:
    return {"coverage": _agg([r["coverage"] for r in rows]), "precision": _agg([r["precision"] for r in rows]),
            "coverage_first_seed": rows[0]["coverage"], "precision_first_seed": rows[0]["precision"],
            "coverage_ci_first_seed": rows[0]["coverage_ci"], "precision_ci_first_seed": rows[0]["precision_ci"],
            "near_dup_vs_recruits": float(np.mean([r["near_dup_vs_recruits"] for r in rows]))}


def aggregate(per_seed: list[dict]) -> dict:
    scr = {}
    for name in ("screened_3", "screened_4"):
        rows = [s["screener"][name] for s in per_seed]
        counts: dict[str, int] = {}
        for r in rows:
            for it in r["items"]:
                counts[it] = counts.get(it, 0) + 1
        scr[name] = {k: _agg([r[k] for r in rows]) for k in ("auc", "accept_rate", "base_rate_missed", "precision",
                                                              "recall")}
        scr[name]["items_chosen_count"] = dict(sorted(counts.items(), key=lambda kv: -kv[1]))
    rec = {}
    for b in BUDGETS:
        rec[str(b)] = {}
        for rname in RECRUITS:
            rows = [s["recruit"][b][rname] for s in per_seed]
            rec[str(b)][rname] = {
                "n_screened": _agg([r["n_screened"] for r in rows]),
                "reached_budget": sum(r["n_recruited"] == b for r in rows),
                "share_missed_among_recruits": _agg([r["share_missed_among_recruits"] for r in rows]),
                **{m: _metric([r[m] for r in rows]) for m in METHODS}}
    paired = []
    for b in BUDGETS:
        for rname in ("screened_3", "screened_4"):
            for m in METHODS:
                d = np.array([s["recruit"][b][rname][m]["coverage"] - s["recruit"][b]["random"][m]["coverage"]
                              for s in per_seed])
                paired.append({"budget": b, "screener": rname, "method": m, "mean_diff": float(d.mean()),
                               "sd": float(d.std(ddof=1)), "wins": int((d > 0).sum()), "n_seeds": len(d)})
    return {"screener": scr, "recruit": rec, "paired": paired}


def run(out_dir: Path = RESULTS) -> dict:
    started = time.perf_counter()
    gss_in = next(d for d in load_inputs() if d.name == "gss")               # holdout already dropped
    _assert_no_holdout(gss_in.real.answers.index, gss_in.splits, "gss real data")
    live = [i.code for i in gss_in.real.items if i.live_ok]
    full_target, full_pool = (gss_in.real.subset(gss_in.splits[s]) for s in ("test", "pool"))
    target, pool = restrict(full_target, live), restrict(full_pool, live)
    base = claude_mode_full(full_target, live)
    ctx = {"pool": pool, "target": target, "base": base, "splits": gss_in.splits,
           "rr": real_distances(target, target.items), "missed": missed_labels(pool, base)}
    per_seed = [run_seed(SEED + i, ctx) for i in range(N_SEEDS)]
    report = {"settings": {"budgets": list(BUDGETS), "n_seeds": N_SEEDS, "screen_sizes": list(SCREEN_SIZES),
                           "accept_target": ACCEPT_TARGET, "k": K, "n_items": len(live), "n_target": len(target.answers),
                           "n_pool": len(pool.answers), "n_boot": N_BOOT, "git_commit": _git_commit(),
                           "pool_missed_share": float(ctx["missed"].mean()),
                           "includes_real_respondents": ["claude_plus_recruits"]},
              **aggregate(per_seed), "runtime_seconds": round(time.perf_counter() - started, 1)}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "screener.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (out_dir / "screener.md").write_text(to_markdown(report), encoding="utf-8")
    return report


# ---------- markdown ----------
def _c(a: dict) -> str:
    return f"{a['mean']:.3f} ({a['p2_5']:.3f}-{a['p97_5']:.3f})"


def _ci(a: dict, key: str) -> str:
    ci, m = a[f"{key}_ci_first_seed"], a[key]
    return (f"{m['mean']:.3f} (seeds {m['p2_5']:.3f}-{m['p97_5']:.3f}); first seed "
            f"{a[f'{key}_first_seed']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")


def to_markdown(r: dict) -> str:
    s = r["settings"]
    lines = ["# Screener targeting simulation (zero cost, GSS test split for scoring)", "",
             f"{s['n_items']} live items, {s['n_seeds']} seeds. Missed = pool person not covered by claude_mode (pool "
             f"5-NN radius; {s['pool_missed_share']:.1%} of the pool). Per seed the pool is split in random halves: one "
             f"half trains the screener (greedy forward selection, CV AUC), the other is the recruiting pool. Threshold "
             f"accepts ~{s['accept_target']:.0%}. Coverage and precision on the test split (n = {s['n_target']}), "
             "`claude_plus_recruits` INCLUDES REAL RESPONDENTS. Coverage and precision cells: mean over seeds (2.5-97.5 "
             f"percentile across seeds); the bootstrap 95% CI belongs to the first seed's point estimate. Commit `{s['git_commit']}`. Runtime {r['runtime_seconds']} s.", "",
             "## Screener quality (evaluated on the other half of the pool)", "",
             "| screener | AUC | accept rate | base rate missed | precision | recall | items chosen (seeds) |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, a in r["screener"].items():
        items = ", ".join(f"{k} {v}" for k, v in list(a["items_chosen_count"].items())[:6])
        lines.append(f"| {name} | {_c(a['auc'])} | {a['accept_rate']['mean']:.2f} | {a['base_rate_missed']['mean']:.2f} | "
                     f"{_c(a['precision'])} | {_c(a['recall'])} | {items} |")
    for m, title in (("copula_on_recruits", "gaussian_copula fit on the recruits"),
                     ("claude_plus_recruits", "claude_mode + recruits swapped in")):
        lines += ["", f"## {title}: test coverage / precision", "",
                  "| budget | recruits | coverage | precision | near-dup |", "| --- | --- | --- | --- | --- |"]
        for b in s["budgets"]:
            for rname in RECRUITS:
                e = r["recruit"][str(b)][rname][m]
                lines.append(f"| {b} | {rname} | {_ci(e, 'coverage')} | {_ci(e, 'precision')} | {e['near_dup_vs_recruits']:.3f} |")
    lines += ["", "## Screening effort and who got recruited", "",
              "| budget | recruits | people screened (mean, range) | budget reached (of 10 seeds) | share missed among recruits |",
              "| --- | --- | --- | --- | --- |"]
    for b in s["budgets"]:
        for rname in RECRUITS:
            e = r["recruit"][str(b)][rname]
            lines.append(f"| {b} | {rname} | {_c(e['n_screened'])} | {e['reached_budget']} | "
                         f"{e['share_missed_among_recruits']['mean']:.2f} |")
    lines += ["", "## Paired: screened minus random (test coverage, same seeds)", "",
              "| budget | screener | method | mean diff | sd | screened wins |", "| --- | --- | --- | --- | --- | --- |"]
    lines += [f"| {d['budget']} | {d['screener']} | {d['method']} | {d['mean_diff']:+.3f} | {d['sd']:.3f} | "
              f"{d['wins']}/{d['n_seeds']} |" for d in r["paired"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    run()
    print((RESULTS / "screener.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
