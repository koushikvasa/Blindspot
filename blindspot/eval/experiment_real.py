"""Experiment REAL: score a real Prolific respondent study (via Flashpoint) on the GSS TEST split.

`python -m blindspot.eval.experiment_real [--csv data/raw/flashpoint/real_prolific_n50.csv] [--out-dir results]`
Works on a partial file too (any n >= 10 usable respondents). No Claude calls (claude_mode comes from the
H3 cache), no Flashpoint calls, holdout never touched.

Crowds, all scored on the GSS test split (20 live items), same metrics as before:
  claude_mode             AI only (budget 0)
  copula_on_recruits      gaussian_copula fit on the real recruits, raw
  copula_raked            same, but recruits are raked (IPF) to the GSS POOL's age band / gender / education /
                          income band shares (weights trimmed to 0.2-5x the mean) and the copula is weighted
  claude_plus_recruits    claude_mode with the recruits swapped in (INCLUDES REAL RESPONDENTS)
  references              gaussian_copula on n random POOL people (mean of 20 draws; n = number of recruits),
                          other_real_people (pool, n = test size)
Diagnostics: recruit composition vs the GSS pool, effective sample size after raking, mean per-item TV between
recruits and the GSS pool (panel / mode effect). The recruits are NOT GSS people: they are never put into pool/test.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from blindspot.agent.guards import near_duplicates
from blindspot.coverage import evaluate, real_distances
from blindspot.data.schema import Respondents
from blindspot.generators.baselines import _nearest_correlation, gaussian_copula, other_real_people
from blindspot.generators.flashpoint_real import DEMO_COLS, SOURCE, load as load_real

from .bakeoff1 import make_crowd
from .experiment0 import K, K_VALUES, N_BOOT, SEED, _assert_no_holdout, _git_commit, load_inputs
from .experiment2 import swap_in_asked
from .experiment_fp import FP_DIR, _agg, claude_mode_full, restrict, tv_distance

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
DEFAULT_CSV = FP_DIR / "real_prolific_n50.csv"
N_REF_DRAWS = 20
WEIGHT_TRIM = (0.2, 5.0)


# ---------- raking and the weighted copula ----------
def rake(demo: pd.DataFrame, targets: dict[str, pd.Series], max_iter: int = 500, tol: float = 1e-9,
         trim: tuple[float, float] = WEIGHT_TRIM) -> np.ndarray:
    """Iterative proportional fitting of person weights to the target shares of each demographic.

    Only categories present in the sample are raked to (targets renormalized over them); people with an
    unknown value in a dimension are left out of that dimension. Weights are trimmed and have mean 1."""
    active = {}
    for col, share in targets.items():
        present = [c for c in share.index if (demo[col] == c).any()]
        if present:
            s = share.loc[present]
            active[col] = s / s.sum()
    w = np.ones(len(demo))
    for _ in range(max_iter):
        gap = 0.0
        for col, tgt in active.items():
            known = demo[col].isin(tgt.index).to_numpy()
            cur = pd.Series(w[known]).groupby(demo[col].to_numpy()[known]).sum()
            cur = cur / cur.sum()
            gap = max(gap, float((cur - tgt).abs().max()))
            factor = (tgt / cur).reindex(demo[col]).to_numpy()
            w = np.where(known, w * np.nan_to_num(factor, nan=1.0), w)
        w = w / w.mean()
        if gap < tol:
            break
    w = np.clip(w, *trim)
    return w / w.mean()


def effective_n(w: np.ndarray) -> float:
    return float(w.sum() ** 2 / (w ** 2).sum())


def weighted_copula(recruits: Respondents, demographics: pd.DataFrame, weights: np.ndarray, seed: int) -> Respondents:
    """Gaussian copula with weighted marginals and a weighted correlation (missing answers filled with the weighted mode)."""
    items, n = recruits.items, len(demographics)
    x = recruits.answers.to_numpy(dtype=float)
    shares, z = [], np.empty_like(x)
    for j, it in enumerate(items):
        col, k = x[:, j].copy(), len(it.options)
        mass = np.bincount(col[~np.isnan(col)].astype(int), weights=weights[~np.isnan(col)], minlength=k)
        col[np.isnan(col)] = float(np.argmax(mass))
        mass = np.bincount(col.astype(int), weights=weights, minlength=k)
        share = mass / mass.sum()
        mid = np.cumsum(share) - share / 2                                  # weighted mid-rank of each option
        z[:, j] = stats.norm.ppf(np.clip(mid[col.astype(int)], 1e-6, 1 - 1e-6))
        shares.append(share)
    cov =np.cov(z, rowvar=False, aweights=weights)
    d = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    corr = _nearest_correlation(cov / np.outer(d, d))
    rng = np.random.default_rng(seed)
    u = stats.norm.cdf(rng.multivariate_normal(np.zeros(len(items)), corr, size=n, method="eigh"))
    cols = [np.minimum(np.searchsorted(np.cumsum(s), u[:, j], side="right"), len(s) - 1) for j, s in enumerate(shares)]
    return make_crowd(np.column_stack(cols).astype(float), demographics, items, "copula_raked")


# ---------- evaluation ----------
def score(target, crowd, rr, seed: int, synthetic, recruits, n_boot: int) -> dict:
    m = evaluate(target, crowd, target.items, k=K, n_boot=n_boot, seed=seed, k_values=K_VALUES, real_dist=rr)
    return {"n_synthetic": m["n_synthetic"], "coverage": m["coverage"], "coverage_ci": m["coverage_ci"],
            "precision": m["precision"], "precision_ci": m["precision_ci"], "rcg_ratio": m["rcg_ratio"],
            "k_sensitivity": m["k_sensitivity"],
            "near_dup_vs_recruits": near_duplicates(synthetic, recruits).share if synthetic is not None else None}


def composition(recruits: Respondents, pool: Respondents) -> dict:
    """Share per category: recruits vs the GSS pool (known values only)."""
    out = {}
    for col in DEMO_COLS:
        pool_share = pool.demographics[col].loc[lambda s: s != "unknown"].value_counts(normalize=True)
        rec_share = recruits.demographics[col].loc[lambda s: s != "unknown"].value_counts(normalize=True)
        out[col] = {c: {"recruits": float(rec_share.get(c, 0.0)), "gss_pool": float(pool_share.get(c, 0.0))}
                    for c in sorted(set(pool_share.index) | set(rec_share.index))}
    return out


def evaluate_recruits(target: Respondents, pool: Respondents, base: Respondents, recruits: Respondents,
                      splits: dict, rr: np.ndarray, seed: int = SEED, n_boot: int = N_BOOT) -> dict:
    """All crowds for one set of real recruits (testable with synthetic stand-ins for target/pool/base)."""
    own = {"pool": list(recruits.answers.index), "test": [], "holdout": []}   # recruits are not GSS people
    targets = {c: pool.demographics[c].loc[lambda s: s != "unknown"].value_counts(normalize=True) for c in DEMO_COLS}
    w = rake(recruits.demographics, targets)
    synthetic = base.subset(list(base.answers.index[len(recruits.answers):]))
    crowds = {
        "claude_mode": (base, None),
        "copula_on_recruits": (gaussian_copula(recruits, target.demographics, own, seed), None),
        "copula_raked": (weighted_copula(recruits, target.demographics, w, seed), None),
        "claude_plus_recruits": (swap_in_asked(base, recruits), synthetic)}
    results = {n: score(target, c, rr, seed, s if s is not None else c, recruits, n_boot) for n, (c, s) in crowds.items()}
    results["claude_mode"]["near_dup_vs_recruits"] = None
    refs = []
    for i in range(N_REF_DRAWS):
        ids = list(np.random.default_rng(seed + i).choice(np.array(pool.answers.index), len(recruits.answers), replace=False))
        crowd = gaussian_copula(pool.subset(ids), target.demographics, splits, seed + i)
        refs.append(score(target, crowd, rr, seed + i, crowd, pool.subset(ids), 20))
    other = other_real_people(pool, len(target.answers), splits, seed)
    return {"crowds": results,
            "references": {f"copula_on_{len(recruits.answers)}_random_pool_people (mean of {N_REF_DRAWS})": {
                "coverage": _agg([r["coverage"] for r in refs]), "precision": _agg([r["precision"] for r in refs])},
                "other_real_people": score(target, other, rr, seed, other, recruits, n_boot)},
            "raking": {"effective_n": effective_n(w), "n": len(w), "max_weight": float(w.max()),
                       "min_weight": float(w.min())},
            "composition": composition(recruits, pool), "mean_item_tv_recruits_vs_gss_pool": tv_distance(recruits, pool)}


def run(csv_path: Path = DEFAULT_CSV, out_dir: Path = RESULTS) -> dict:
    started = time.perf_counter()
    gss_in = next(d for d in load_inputs() if d.name == "gss")               # holdout already dropped
    _assert_no_holdout(gss_in.real.answers.index, gss_in.splits, "gss real data")
    live = [i.code for i in gss_in.real.items if i.live_ok]
    full_target, full_pool = (gss_in.real.subset(gss_in.splits[s]) for s in ("test", "pool"))
    target, pool = restrict(full_target, live), restrict(full_pool, live)
    recruits, load_report = load_real(csv_path, target.items)
    base = claude_mode_full(full_target, live)
    res = evaluate_recruits(target, pool, base, recruits, gss_in.splits, real_distances(target, target.items))
    report = {"settings": {"source": SOURCE, "n_recruits": len(recruits.answers), "n_target": len(target.answers),
                           "n_items": len(live), "k": K, "n_boot": N_BOOT, "n_ref_draws": N_REF_DRAWS,
                           "weight_trim": list(WEIGHT_TRIM), "partial": len(recruits.answers) < 50,
                           "git_commit": _git_commit(), "includes_real_respondents": ["claude_plus_recruits"]},
              "load": load_report.to_dict(), **res, "runtime_seconds": round(time.perf_counter() - started, 1)}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "experiment_real.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (out_dir / "experiment_real.md").write_text(to_markdown(report), encoding="utf-8")
    return report


# ---------- markdown ----------
def _ci(m: dict, key: str) -> str:
    ci = m[f"{key}_ci"]
    return f"{m[key]:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def to_markdown(r: dict) -> str:
    s, ld, rk = r["settings"], r["load"], r["raking"]
    lines = ["# Experiment REAL: Prolific respondents via Flashpoint, scored on the GSS test split", "",
             f"Recruits n = {s['n_recruits']}" + (" (PARTIAL file)" if s["partial"] else "") + f"; file had "
             f"{ld['n_rows_in_file']} rows, {ld['n_complete']} complete, {ld['n_quality_excluded']} quality-excluded. "
             f"{s['n_items']} live items, test n = {s['n_target']}, k = {s['k']}, {s['n_boot']} bootstrap resamples. "
             "`claude_plus_recruits` INCLUDES REAL RESPONDENTS. Commit "
             f"`{s['git_commit']}`. Runtime {r['runtime_seconds']} s.", "",
             "| crowd | n | coverage [95% CI] | precision [95% CI] | near-dup vs recruits |",
             "| --- | --- | --- | --- | --- |"]
    for name, m in sorted(r["crowds"].items(), key=lambda kv: -kv[1]["coverage"]):
        nd = "-" if m["near_dup_vs_recruits"] is None else f"{m['near_dup_vs_recruits']:.3f}"
        lines.append(f"| {name} | {m['n_synthetic']} | {_ci(m, 'coverage')} | {_ci(m, 'precision')} | {nd} |")
    for name, v in r["references"].items():
        if "coverage" in v and isinstance(v["coverage"], dict):
            lines.append(f"| ref: {name} | - | {v['coverage']['mean']:.3f} (sd {v['coverage']['sd']:.3f}) | "
                         f"{v['precision']['mean']:.3f} | - |")
        else:
            lines.append(f"| ref: {name} | {v['n_synthetic']} | {_ci(v, 'coverage')} | {_ci(v, 'precision')} | - |")
    lines += ["", "k-sensitivity (coverage / precision):", "", "| crowd | k=3 | k=5 | k=10 |", "| --- | --- | --- | --- |"]
    for name, m in r["crowds"].items():
        lines.append(f"| {name} | " + " | ".join(f"{m['k_sensitivity'][k]['coverage']:.3f} / {m['k_sensitivity'][k]['precision']:.3f}"
                                                  for k in ("3", "5", "10")) + " |")
    lines += ["", f"Raking to the GSS pool: effective n {rk['effective_n']:.1f} of {rk['n']}, weights "
              f"{rk['min_weight']:.2f}-{rk['max_weight']:.2f} (trimmed to {s['weight_trim'][0]}-{s['weight_trim'][1]}x mean). "
              f"Mean per-item TV between recruits and the GSS pool: {r['mean_item_tv_recruits_vs_gss_pool']:.3f}.", "",
              "## Recruit composition vs GSS pool", "", "| variable | category | recruits | GSS pool |", "| --- | --- | --- | --- |"]
    for col, cats in r["composition"].items():
        lines += [f"| {col} | {c} | {v['recruits']:.2f} | {v['gss_pool']:.2f} |" for c, v in cats.items()]
    inv = ld["answers"]
    lines += ["", f"Answer mapping: {inv['n_answers']} answers, {inv['missing']} missing, {inv['unmapped']} unmapped "
              f"({inv['invalid_share']:.2%} invalid). Unmapped demographics: {ld['demographics_unmapped'] or 'none'}."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a real Flashpoint/Prolific study on the GSS test split")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-dir", type=Path, default=RESULTS)
    args = parser.parse_args()
    run(args.csv, args.out_dir)
    print((args.out_dir / "experiment_real.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
