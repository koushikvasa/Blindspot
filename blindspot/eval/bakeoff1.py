"""Bake-off round 1 (zero cost): select-from-many and a Twin model mix. TEST split targets, never holdout.

`python -m blindspot.eval.bakeoff1`
No Claude and no Flashpoint calls: candidates come from the cached H3 per-item probabilities and the
saved Flashpoint rows. The pool split is used only as a realism FILTER (distance of a candidate to its
5 nearest pool people); no pool row is inserted into any crowd. Near-duplicate share = crowd people
within 0.02 Gower of any pool person.

Part A  GSS (20 live items): per test persona draw 50 candidates from claude_sampled probabilities and
        greedily keep the one minimising realism - lambda * novelty (novelty = distance to the nearest
        already selected member), lambda in {0, .5, 1, 2}; 5 seeds.
Part B  the same score choosing 200 of the 400 saved Flashpoint people; compared at n=200.
Part C  Twin (30 items): 50% claude_mode + 50% gpt41mini_twin (10 seeds) and the union upper bound.
"""
import json
import time
from pathlib import Path

import numpy as np

from blindspot.agent.guards import near_duplicates
from blindspot.coverage import evaluate, real_distances
from blindspot.data import gss
from blindspot.data.schema import Respondents
from blindspot.generators import claude_personas as cp
from blindspot.generators.baselines import gaussian_copula, independent_marginals, other_real_people
from blindspot.generators.claude_client import ResponseCache, Usage
from blindspot.generators.flashpoint_panel import load as load_flashpoint
from blindspot.space import encode, gower

from .experiment0 import K, K_VALUES, N_BOOT, SEED, _assert_no_holdout, _git_commit, load_inputs
from .experiment_fp import FP_DIR, RUNS, _agg, restrict
from .splits import load_splits

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
LAMBDAS = (0.0, 0.5, 1.0, 2.0)
N_CANDIDATES = 50
REALISM_K = 5
N_SEEDS = 5
MIX_SEEDS = 10
N_FP_SELECT = 200


# ---------- selection ----------
def _scale(items) -> np.ndarray:
    return np.array([max(len(it.options) - 1, 1) for it in items], dtype=float)


def realism(cands_raw: np.ndarray, pool: Respondents) -> np.ndarray:
    """Mean Gower distance to the REALISM_K nearest pool people (lower = more realistic)."""
    d = np.nan_to_num(gower(cands_raw / _scale(pool.items), encode(pool), pool.items), nan=1.0)
    return np.partition(d, REALISM_K - 1, axis=1)[:, :REALISM_K].mean(axis=1)


def _novelty(cand_enc: np.ndarray, chosen_enc: np.ndarray, items) -> np.ndarray:
    if len(chosen_enc) == 0:
        return np.zeros(len(cand_enc))
    return np.nan_to_num(gower(cand_enc, chosen_enc, items), nan=1.0).min(axis=1)


def select_per_persona(cands: list[np.ndarray], real: list[np.ndarray], lam: float, items) -> np.ndarray:
    """For each persona (in order) keep the candidate minimising realism - lam * novelty."""
    scale, chosen = _scale(items), []
    for c, r in zip(cands, real):
        nov = _novelty(c / scale, np.array(chosen) / scale if chosen else np.empty((0, len(items))), items)
        chosen.append(c[int(np.argmin(r - lam * nov))])
    return np.array(chosen)


def select_from_pool(cands: np.ndarray, real: np.ndarray, lam: float, n: int, items) -> np.ndarray:
    """Greedily choose n of the candidates (no repeats); returns the chosen row indices."""
    enc = cands / _scale(items)
    nov = np.zeros(len(cands))
    taken = np.zeros(len(cands), bool)
    picks = []
    for _ in range(n):
        score = np.where(taken, np.inf, real - lam * nov)
        j = int(np.argmin(score))
        picks.append(j)
        taken[j] = True
        d = np.nan_to_num(gower(enc, enc[j:j + 1], items), nan=1.0)[:, 0]
        nov = d if len(picks) == 1 else np.minimum(nov, d)
    return np.array(picks)


def draw_candidates(parsed: list, items, seed: int) -> list[np.ndarray]:
    """50 candidates per persona, every item drawn independently from that persona's probabilities."""
    out = []
    for i, x in enumerate(parsed):
        rng = np.random.default_rng([seed, i])
        cols = [rng.choice(len(it.options), size=N_CANDIDATES, p=x.probs[it.code]).astype(float)
                if it.code in x.probs else np.full(N_CANDIDATES, np.nan) for it in items]
        out.append(np.column_stack(cols))
    return out


def cached_parsed(plan: cp.Plan, cache: ResponseCache) -> list:
    """Parsed answers incl. probabilities for every persona, from the cache only (as cp.execute merges retries)."""
    parsed = [cp.parse_record(cache.get(s.key), plan.items) for s in plan.specs]
    for i, x in enumerate(parsed):
        if not x.complete(plan.items):
            rec = cache.get(plan.retry_spec(i).key)
            if rec is not None:
                parsed[i] = cp.merge(x, cp.parse_record(rec, plan.items))
    return parsed


# ---------- evaluation ----------
def make_crowd(raw: np.ndarray, demo, items, name: str) -> Respondents:
    ids = [f"{name}_{i:05d}" for i in range(len(raw))]
    import pandas as pd
    return Respondents(pd.DataFrame(raw, index=ids, columns=[it.code for it in items]),
                       demo.reset_index(drop=True).set_axis(ids) if demo is not None else
                       pd.DataFrame(index=ids), items, f"bakeoff:{name}")


def eval_crowd(target: Respondents, crowd: Respondents, rr, pool: Respondents, seed: int, n_boot: int = N_BOOT) -> dict:
    m = evaluate(target, crowd, target.items, k=K, n_boot=n_boot, seed=seed, k_values=K_VALUES, real_dist=rr)
    out = {k: v for k, v in m.items() if not k.startswith("_") and k not in ("uncovered_ids", "per_family_entropy")}
    out["near_dup_share_vs_pool"] = near_duplicates(crowd, pool).share
    out["_covered"] = m["_real_covered"]
    return out


def aggregate(runs: list[dict]) -> dict:
    """Mean and range across seeds, plus the bootstrap CI of the first seed."""
    def agg(f):
        return _agg([f(r) for r in runs]) if len(runs) > 1 else {"mean": f(runs[0]), "sd": 0.0,
                                                                   "p2_5": f(runs[0]), "p97_5": f(runs[0])}
    out = {"n_seeds": len(runs), "n_synthetic": runs[0]["n_synthetic"],
           "coverage": agg(lambda r: r["coverage"]), "precision": agg(lambda r: r["precision"]),
           "near_dup_share_vs_pool": agg(lambda r: r["near_dup_share_vs_pool"]),
           "coverage_ci_first_seed": runs[0]["coverage_ci"], "precision_ci_first_seed": runs[0]["precision_ci"],
           "seed_min_max_coverage": [min(r["coverage"] for r in runs), max(r["coverage"] for r in runs)],
           "k_sensitivity": {kk: {s: float(np.mean([r["k_sensitivity"][kk][s] for r in runs]))
                                  for s in ("coverage", "precision")} for kk in runs[0]["k_sensitivity"]}}
    return out


# ---------- parts ----------
def part_a(target, pool, rr, parsed, items, splits, claude_crowds) -> dict:
    """Select-from-many on GSS, n = 334, 20 live items."""
    rows = {"claude_mode": aggregate([eval_crowd(target, claude_crowds["mode"], rr, pool, SEED)]),
            "claude_sampled": aggregate([eval_crowd(target, claude_crowds["sampled"], rr, pool, SEED)]),
            "gaussian_copula": aggregate([eval_crowd(target, gaussian_copula(pool, target.demographics, splits, SEED),
                                                     rr, pool, SEED)]),
            "independent_marginals": aggregate([eval_crowd(
                target, independent_marginals(pool, target.demographics, splits, SEED), rr, pool, SEED)]),
            "other_real_people": aggregate([eval_crowd(
                target, other_real_people(pool, len(target.answers), splits, SEED), rr, pool, SEED)])}
    per_lam: dict[float, list[dict]] = {lam: [] for lam in LAMBDAS}
    for s in range(N_SEEDS):
        seed = SEED + s
        cands = draw_candidates(parsed, items, seed)
        reals = [realism(c, pool) for c in cands]
        for lam in LAMBDAS:
            raw = select_per_persona(cands, reals, lam, items)
            per_lam[lam].append(eval_crowd(target, make_crowd(raw, target.demographics, items, f"sel{lam}"),
                                           rr, pool, seed))
    rows.update({f"select_from_50_lambda_{lam}": aggregate(v) for lam, v in per_lam.items()})
    return rows


def part_b(target, pool, rr, items, splits) -> dict:
    """Same score choosing 200 of the 400 saved Flashpoint people; evaluated at n = 200."""
    meta = json.loads((FP_DIR / "runs_meta.json").read_text(encoding="utf-8"))
    fps = [load_flashpoint(FP_DIR / meta["runs"][t]["rows_file"], items, t)[0] for t in RUNS]
    raw = np.vstack([f.answers.to_numpy() for f in fps])
    real = realism(raw, pool)
    rows = {f"flashpoint_{t}": aggregate([eval_crowd(target, f, rr, pool, SEED)]) for t, f in zip(RUNS, fps)}
    both = make_crowd(raw, None, items, "fp400")
    rand = []
    for s in range(20):
        idx = np.sort(np.random.default_rng(SEED + s).choice(len(raw), N_FP_SELECT, replace=False))
        rand.append(eval_crowd(target, make_crowd(raw[idx], None, items, "fprand"), rr, pool, SEED + s, 20))
    rows["flashpoint_random_200_of_400"] = aggregate(rand)
    rows["flashpoint_all_400 (n=400, subsampled to 334 by evaluate)"] = aggregate([eval_crowd(target, both, rr, pool, SEED)])
    for lam in LAMBDAS:
        picks = select_from_pool(raw, real, lam, N_FP_SELECT, items)
        rows[f"flashpoint_select_200_of_400_lambda_{lam}"] = aggregate(
            [eval_crowd(target, make_crowd(raw[picks], None, items, f"fpsel{lam}"), rr, pool, SEED)])
    ref = json.loads((RESULTS / "experiment_fp.json").read_text(encoding="utf-8"))["baselines_at_200"]
    rows["_reference_experiment_fp_n200"] = {n: {"coverage_mean": v["coverage"]["mean"],
                                                 "precision_mean": v["precision"]["mean"]} for n, v in ref.items()}
    return rows


def part_c(twin_input) -> dict:
    """Twin: 50% claude_mode + 50% gpt41mini_twin, and the union upper bound on coverage."""
    from dotenv import dotenv_values

    d = twin_input
    _assert_no_holdout(d.real.answers.index, d.splits, "twin real data")
    target = d.real.subset(d.splits["test"])
    pool = d.real.subset(d.splits["pool"])
    rr = real_distances(target, target.items)
    model = dotenv_values(ROOT / ".env")["CLAUDE_MODEL"]
    plan = cp.plan("twin2k", target.demographics, target.items, model)
    claude = cp.execute(plan, None, ResponseCache(), Usage(batch=True), seed=SEED, batch=True).mode
    twin = d.fixed_crowds["gpt41mini_twin"]
    c_eval = eval_crowd(target, claude, rr, pool, SEED)
    t_eval = eval_crowd(target, twin, rr, pool, SEED)
    runs = []
    for s in range(MIX_SEEDS):
        pick_claude = np.random.default_rng(SEED + s).permutation(len(target.answers)) < len(target.answers) // 2
        raw = np.where(pick_claude[:, None], claude.answers.to_numpy(), twin.answers.to_numpy())
        runs.append(eval_crowd(target, make_crowd(raw, target.demographics, target.items, "mix"), rr, pool, SEED + s))
    union = np.logical_or(c_eval["_covered"], t_eval["_covered"])
    ref = json.loads((RESULTS / "experiment1.json").read_text(encoding="utf-8"))["datasets"]["twin2k"]["crowds"]
    return {"n_target": len(target.answers), "n_items": len(target.items),
            "claude_mode": aggregate([c_eval]), "gpt41mini_twin": aggregate([t_eval]),
            "mix_50_50": aggregate(runs), "union_upper_bound_coverage": float(union.mean()),
            "reference_experiment1": {n: {"coverage": ref[n]["coverage"], "precision": ref[n]["precision"]}
                                      for n in ("gaussian_copula", "other_real_people", "independent_marginals")}}


def run(out_dir: Path = RESULTS) -> dict:
    started = time.perf_counter()
    inputs = {d.name: d for d in load_inputs()}               # holdout already dropped
    gss_in = inputs["gss"]
    live = [i.code for i in gss_in.real.items if i.live_ok]
    full_target, full_pool = (gss_in.real.subset(gss_in.splits[s]) for s in ("test", "pool"))
    target, pool = restrict(full_target, live), restrict(full_pool, live)
    rr = real_distances(target, target.items)
    from dotenv import dotenv_values

    model = dotenv_values(ROOT / ".env")["CLAUDE_MODEL"]
    plan = cp.plan("gss", full_target.demographics, full_target.items, model)
    cache = ResponseCache()
    crowds = cp.execute(plan, None, cache, Usage(batch=True), seed=SEED, batch=True)
    claude = {"mode": restrict(crowds.mode, live), "sampled": restrict(crowds.sampled, live)}
    parsed = cached_parsed(plan, cache)
    report = {
        "settings": {"lambdas": list(LAMBDAS), "n_candidates": N_CANDIDATES, "realism_k": REALISM_K,
                     "n_seeds": N_SEEDS, "mix_seeds": MIX_SEEDS, "k": K, "k_values": list(K_VALUES),
                     "n_boot": N_BOOT, "target_split": "test", "items_gss": live, "n_target_gss": len(target.answers),
                     "near_dup_threshold": 0.02, "git_commit": _git_commit(),
                     "note": "GSS parts use the 20 live items (H3's 24-item numbers differ). Pool used only "
                             "as a realism filter."},
        "A_gss_select_from_many_n334": part_a(target, pool, rr, parsed, target.items, gss_in.splits, claude),
        "B_flashpoint_select_n200": part_b(target, pool, rr, target.items, gss_in.splits),
        "C_twin_mix": part_c(inputs["twin2k"]),
    }
    report["runtime_seconds"] = round(time.perf_counter() - started, 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bakeoff1.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (out_dir / "bakeoff1.md").write_text(to_markdown(report), encoding="utf-8")
    return report


# ---------- markdown ----------
def _row(name: str, a: dict) -> str:
    cov, prec = a["coverage"], a["precision"]
    ci, pci = a["coverage_ci_first_seed"], a["precision_ci_first_seed"]
    rng = (f"; seeds {a['seed_min_max_coverage'][0]:.3f}-{a['seed_min_max_coverage'][1]:.3f}"
           if a["n_seeds"] > 1 else "")
    ks = " | ".join(f"{a['k_sensitivity'][k]['coverage']:.3f} / {a['k_sensitivity'][k]['precision']:.3f}"
                    for k in ("3", "5", "10"))
    return (f"| {name} | {a['n_synthetic']} | {cov['mean']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]{rng} | "
            f"{prec['mean']:.3f} [{pci[0]:.3f}, {pci[1]:.3f}] | {a['near_dup_share_vs_pool']['mean']:.3f} | {ks} |")


HEADER = ["| crowd | n | coverage [95% CI, first seed]; seed range | precision [95% CI] | near-dup vs pool | "
          "k=3 cov/prec | k=5 | k=10 |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]


def to_markdown(r: dict) -> str:
    s = r["settings"]
    lines = ["# Bake-off round 1: select-from-many and model mix (zero cost, test split only)", "",
             f"No Claude or Flashpoint calls. Candidates: {s['n_candidates']} per persona from cached claude_sampled "
             f"probabilities; realism = mean Gower to the {s['realism_k']} nearest pool people; novelty = distance to the "
             f"nearest already selected member; pick = argmin realism - lambda*novelty. {s['n_seeds']} seeds for "
             f"selection rows (mean; seed range), bootstrap CI from the first seed. Near-dup = share within "
             f"{s['near_dup_threshold']} Gower of any pool person. Commit `{s['git_commit']}`. "
             f"Runtime {r['runtime_seconds']} s.", "", s["note"], "",
             f"## A. GSS select-from-many (target n = {s['n_target_gss']}, 20 live items)", ""] + HEADER
    lines += [_row(n, a) for n, a in r["A_gss_select_from_many_n334"].items()]
    lines += ["", "## B. Flashpoint candidates, choose 200 of 400 (n = 200)", ""] + HEADER
    b = r["B_flashpoint_select_n200"]
    lines += [_row(n, a) for n, a in b.items() if not n.startswith("_")]
    ref = b["_reference_experiment_fp_n200"]
    lines += ["", "Reference at n = 200 (mean of 20 draws, from experiment_fp): " + "; ".join(
        f"{n} {v['coverage_mean']:.3f} / {v['precision_mean']:.3f}" for n, v in ref.items()), ""]
    c = r["C_twin_mix"]
    lines += [f"## C. Twin model mix (target n = {c['n_target']}, {c['n_items']} items)", ""] + HEADER
    lines += [_row(n, c[n]) for n in ("claude_mode", "gpt41mini_twin", "mix_50_50")]
    lines += ["", f"Union upper bound (real person covered if either full crowd covers them): coverage "
              f"**{c['union_upper_bound_coverage']:.3f}**. Reference (experiment1): " + "; ".join(
                  f"{n} {v['coverage']:.3f} / {v['precision']:.3f}" for n, v in c["reference_experiment1"].items())]
    return "\n".join(lines) + "\n"


def main() -> None:
    run()
    print((RESULTS / "bakeoff1.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
