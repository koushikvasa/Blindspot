"""FINAL HOLDOUT RUN. Run ONCE. Never rerun on the holdout.

`python -m blindspot.eval.final_holdout --estimate-only`   cost estimate for the holdout personas, then stop
`python -m blindspot.eval.final_holdout`                   generate Claude crowds (cost-guarded), then score
`python -m blindspot.eval.final_holdout --split test --out-dir <scratch>`   rehearsal on the TEST split

Target = HOLDOUT split (Twin n=412, GSS n=224). Every fit, filter and draw uses the POOL split only. Claude
crowds for the holdout personas are generated from their DEMOGRAPHICS only (holdout answers never enter a
prompt); the generation stage can be retried, scoring cannot: `results/final_holdout.started` is written
before the first holdout answer is scored and the script refuses to start if it (or the results) exist.
Twin: all 30 items. GSS: the 20 live items. No fairness table (holdout too small). The git commit hash is
recorded in the output.
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from blindspot.agent.guards import near_duplicates
from blindspot.coverage import evaluate, real_distances, subsample
from blindspot.data import gss, twin2k
from blindspot.data.schema import Respondents
from blindspot.generators import claude_personas as cp
from blindspot.generators.baselines import gaussian_copula, independent_marginals, other_real_people
from blindspot.generators.claude_client import (DEFAULT_COST_LIMIT_USD, CostLimitExceeded, ResponseCache, Usage)
from blindspot.generators.flashpoint_panel import load as load_flashpoint

from .bakeoff1 import HEADER, _row, aggregate, cached_parsed, draw_candidates, make_crowd, realism, select_per_persona
from .experiment0 import K, K_VALUES, N_BOOT, SEED, _git_commit
from .experiment1 import bunching_share, guard
from .experiment_fp import FP_DIR, RUNS, restrict
from .splits import load_splits

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
MARKER = "final_holdout.started"
MAX_COST_USD = 3.0
MIX_SEEDS = 10
SELECT_SEEDS = 5
ASKED_SEEDS = 10
ASKED_BUDGETS = (25, 50)
SELECT_LAMBDA = 0.5


# ---------- data ----------
def load_world(split: str) -> dict:
    """target = `split`, pool = pool; nothing else is used. Items: Twin all 30, GSS the 20 live ones."""
    tw, g = twin2k.load(), gss.load()
    t_sp, g_sp = load_splits("twin2k"), load_splits("gss")
    live = [i.code for i in g.items if i.live_ok]
    return {
        "twin2k": {"splits": t_sp, "full_target": tw["real"].subset(t_sp[split]),
                   "pool": tw["real"].subset(t_sp["pool"]), "gpt": tw["gpt41mini_twin"].subset(t_sp[split]),
                   "retest": tw["retest"].subset(t_sp[split]), "codes": [i.code for i in tw["real"].items]},
        "gss": {"splits": g_sp, "full_target": g.subset(g_sp[split]), "pool": g.subset(g_sp["pool"]), "codes": live}}


def make_plans(world: dict, model: str) -> dict[str, cp.Plan]:
    return {n: cp.plan(n, w["full_target"].demographics, w["full_target"].items, model) for n, w in world.items()}


# ---------- evaluation helpers ----------
def eval_one(target: Respondents, crowd: Respondents, rr, pool: Respondents, seed: int, n_boot: int = N_BOOT) -> dict:
    m = evaluate(target, crowd, target.items, k=K, n_boot=n_boot, seed=seed, k_values=K_VALUES, real_dist=rr)
    out = {k: v for k, v in m.items() if not k.startswith("_") and k != "uncovered_ids"}
    out["near_dup_share_vs_pool"] = near_duplicates(crowd, pool).share
    out["bunching_gt_80"] = bunching_share(crowd)
    return out


def single(target, crowd, rr, pool, seed=SEED) -> dict:
    r = eval_one(target, crowd, rr, pool, seed)
    return {**aggregate([r]), "entropy": r["per_family_entropy"]["crowd"], "bunching_gt_80": r["bunching_gt_80"],
            "real_entropy": r["per_family_entropy"]["real"]}


def many(target, crowds_by_seed, rr, pool, n_boot=20) -> dict:
    return aggregate([eval_one(target, c, rr, pool, SEED + i, n_boot) for i, c in enumerate(crowds_by_seed)])


# ---------- Twin ----------
def score_twin(w: dict, claude, rr_cache: dict) -> dict:
    target, pool, splits = w["target"], w["pool"], w["splits"]
    rr = real_distances(target, target.items)
    demo = target.demographics
    crowds = {"retest": w["retest"], "other_real_people": other_real_people(pool, len(target.answers), splits, SEED),
              "gaussian_copula": gaussian_copula(pool, demo, splits, SEED),
              "independent_marginals": independent_marginals(pool, demo, splits, SEED),
              "claude_mode": claude["mode"], "gpt41mini_twin": w["gpt"], "claude_sampled": claude["sampled"]}
    rows = {n: single(target, c, rr, pool) for n, c in crowds.items()}
    mixes = []
    for i in range(MIX_SEEDS):
        half = np.random.default_rng(SEED + i).permutation(len(target.answers)) < len(target.answers) // 2
        raw = np.where(half[:, None], claude["mode"].answers.to_numpy(), w["gpt"].answers.to_numpy())
        mixes.append(make_crowd(raw, demo, target.items, "mix"))
    rows["mix_50_50_claude_mode_gpt41mini"] = many(target, mixes, rr, pool)
    return {"n_target": len(target.answers), "n_pool": len(pool.answers), "n_items": len(target.items),
            "rows": rows, "real_entropy": rows["claude_mode"]["real_entropy"]}


# ---------- GSS ----------
def score_gss(w: dict, claude, parsed) -> dict:
    target, pool, splits = w["target"], w["pool"], w["splits"]
    items, rr, demo = target.items, real_distances(target, target.items), target.demographics
    crowds = {"claude_mode": claude["mode"], "claude_sampled": claude["sampled"],
              "independent_marginals": independent_marginals(pool, demo, splits, SEED),
              "gaussian_copula": gaussian_copula(pool, demo, splits, SEED),
              "other_real_people": other_real_people(pool, len(target.answers), splits, SEED)}
    rows = {n: single(target, c, rr, pool) for n, c in crowds.items()}
    sel = []
    for s in range(SELECT_SEEDS):
        cands = draw_candidates(parsed, items, SEED + s)
        raw = select_per_persona(cands, [realism(c, pool) for c in cands], SELECT_LAMBDA, items)
        sel.append(make_crowd(raw, demo, items, "select"))
    rows[f"select_from_50_lambda_{SELECT_LAMBDA}_pool_filter"] = many(target, sel, rr, pool, N_BOOT)
    ids = np.array(pool.answers.index)
    for b in ASKED_BUDGETS:
        cops = []
        for s in range(ASKED_SEEDS):
            asked = pool.subset(list(ids[np.random.default_rng(SEED + s).permutation(len(ids))[:b]]))
            cops.append(gaussian_copula(asked, demo, splits, SEED + s))
        rows[f"copula_on_asked_B{b}"] = many(target, cops, rr, pool)
    meta = json.loads((FP_DIR / "runs_meta.json").read_text(encoding="utf-8"))
    for t in RUNS:
        fp = load_flashpoint(FP_DIR / meta["runs"][t]["rows_file"], items, t)[0]
        rows[f"flashpoint_{t}_n200"] = single(target, fp, rr, pool)
    return {"n_target": len(target.answers), "n_pool": len(pool.answers), "n_items": len(items), "rows": rows,
            "real_entropy": rows["claude_mode"]["real_entropy"]}


# ---------- run ----------
def claude_pair(plan: cp.Plan, client, cache, usage, batch: bool, limit: float, codes: list[str]):
    crowds = cp.execute(plan, client, cache, usage, seed=SEED, batch=batch, cost_limit=limit)
    return {"mode": restrict(crowds.mode, codes), "sampled": restrict(crowds.sampled, codes),
            "failures": len(crowds.failures)}


def run(split: str, out_dir: Path, estimate_only: bool, max_cost: float, batch: bool = True) -> dict | None:
    started = time.perf_counter()
    final = split == "holdout"
    if final and ((out_dir / "final_holdout.json").exists() or (out_dir / MARKER).exists()):
        raise SystemExit("REFUSING: the holdout was already scored (marker or results exist). Never rerun it.")
    from dotenv import dotenv_values, load_dotenv

    load_dotenv(ROOT / ".env")
    model = dotenv_values(ROOT / ".env")["CLAUDE_MODEL"]
    world = load_world(split)
    plans, cache, usage = make_plans(world, model), ResponseCache(), Usage(batch=batch)
    client = None
    if any(cache.get(s.key) is None for p in plans.values() for s in p.specs):
        import anthropic
        client = anthropic.Anthropic()
    est = guard(plans, client, cache, max_cost, batch) if client else {"total": {"calls": 0, "cost_usd": 0.0}}
    print(f"[estimate] {split}: {est['total']['calls']} uncached calls, ~${est['total']['cost_usd']:.2f} "
          f"(limit ${max_cost:.2f})")
    if estimate_only:
        return {"estimate": est}
    claude = {n: claude_pair(plans[n], client, cache, usage, batch, max_cost, world[n]["codes"]) for n in plans}
    parsed = cached_parsed(plans["gss"], cache)
    if final:                                           # point of no return: holdout answers get scored below
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / MARKER).write_text(f"started {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
                                      f"commit {_git_commit()}\n", encoding="utf-8")
    for n, w in world.items():
        w["target"], w["pool"] = restrict(w["full_target"], w["codes"]), restrict(w["pool"], w["codes"])
    if "gpt" in world["twin2k"]:
        world["twin2k"]["gpt"] = restrict(world["twin2k"]["gpt"], world["twin2k"]["codes"])
        world["twin2k"]["retest"] = restrict(world["twin2k"]["retest"], world["twin2k"]["codes"])
    report = {
        "settings": {"split": split, "k": K, "k_values": list(K_VALUES), "n_boot": N_BOOT, "seed": SEED,
                     "git_commit": _git_commit(), "model": model, "batch": batch, "python": platform.python_version(),
                     "numpy": np.__version__, "mix_seeds": MIX_SEEDS, "select_seeds": SELECT_SEEDS,
                     "asked_seeds": ASKED_SEEDS, "select_lambda": SELECT_LAMBDA, "fit_split": "pool",
                     "claude_note": "holdout personas generated from demographics only, same prompt/cache as H3; "
                                    "GSS prompts list all 24 items, scored on the 20 live ones"},
        "cost": {"estimate": est["total"], "actual": usage.to_dict()},
        "claude_failures": {n: c["failures"] for n, c in claude.items()},
        "twin2k": score_twin(world["twin2k"], claude["twin2k"], {}),
        "gss": score_gss(world["gss"], claude["gss"], parsed),
        "runtime_seconds": round(time.perf_counter() - started, 1)}
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "final_holdout" if final else "final_holdout_rehearsal_test_split"
    (out_dir / f"{stem}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (out_dir / f"{stem}.md").write_text(to_markdown(report), encoding="utf-8")
    return report


# ---------- markdown ----------
def _entropy_table(ds: dict) -> list[str]:
    names = [n for n, r in ds["rows"].items() if "entropy" in r]
    lines = ["| family | real target | " + " | ".join(names) + " |", "| --- |" + " --- |" * (len(names) + 1)]
    for fam, h in ds["real_entropy"].items():
        lines.append(f"| {fam} | {h:.2f} | " + " | ".join(f"{ds['rows'][n]['entropy'][fam]:.2f}" for n in names) + " |")
    return lines


def to_markdown(r: dict) -> str:
    s, c = r["settings"], r["cost"]["actual"]
    lines = [f"# Final holdout run ({s['split']} split)", "",
             f"Target = {s['split']} split; every fit, filter and draw on the POOL split only. Commit `{s['git_commit']}`. "
             f"Model `{s['model']}`, Claude holdout personas from demographics only. k = {s['k']}, {s['n_boot']} bootstrap "
             f"resamples, seed {s['seed']}. Multi-seed rows: mean over seeds, bootstrap CI from the first seed, seed range "
             f"in the coverage cell (mix {s['mix_seeds']} seeds, select {s['select_seeds']}, asked {s['asked_seeds']}). "
             f"Cost: estimate ${r['cost']['estimate']['cost_usd']:.2f}, actual ${c['cost_usd']:.2f} for {c['calls']} "
             f"calls. Runtime {r['runtime_seconds']} s. Claude people with a missing item after retry: {r['claude_failures']}.", ""]
    for key, title in (("twin2k", "Twin-2K-500"), ("gss", "GSS 2024, 20 live items")):
        ds = r[key]
        lines += [f"## {title} (target n = {ds['n_target']}, pool n = {ds['n_pool']}, {ds['n_items']} items)", ""] + HEADER
        lines += [_row(n, a) for n, a in ds["rows"].items()]
        lines += ["", "Per-family mean normalized entropy:", ""] + _entropy_table(ds)
        lines += ["", "Share of items with more than 80% of people on one option: " + ", ".join(
            f"{n} {a['bunching_gt_80']:.2f}" for n, a in ds["rows"].items() if "bunching_gt_80" in a), ""]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="FINAL holdout run (once)")
    parser.add_argument("--split", choices=("holdout", "test"), default="holdout")
    parser.add_argument("--out-dir", type=Path, default=RESULTS)
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--max-cost", type=float, default=MAX_COST_USD)
    args = parser.parse_args()
    try:
        out = run(args.split, args.out_dir, args.estimate_only, args.max_cost)
    except CostLimitExceeded as e:
        raise SystemExit(f"STOPPED by cost guard: {e}") from None
    if args.estimate_only:
        print(json.dumps(out["estimate"], indent=1))
        return
    print((args.out_dir / ("final_holdout.md" if args.split == "holdout" else
                           "final_holdout_rehearsal_test_split.md")).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
