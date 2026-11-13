"""Experiment 1b: does a bigger model reduce the coverage gap? TEST split only, the holdout is never loaded for scoring.

`python -m blindspot.eval.experiment1b --estimate-only`   cost estimate per model, then stop
`python -m blindspot.eval.experiment1b`                    run (cost-guarded, $5 per single run), score, write results

Same H3 generator (claude_personas.py, same prompts, demographics only, same item sets). The model is a parameter:
Haiku 5.5 comes from the H3 cache (CLAUDE_MODEL), Sonnet 5.5 runs on GSS and Twin test, Opus 5.5 on GSS test only.
For each model: claude_mode and claude_sampled scored with the H2 metrics; on GSS also on the 20 live items, plus
select_from_50 (lambda 0 and 0.5, pool filter) from that model's cached probabilities (no extra cost).
References: shuffled answers, copula on the pool, other real people, copula on 25 random pool people (10 seeds).
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from blindspot.agent.guards import near_duplicates
from blindspot.coverage import evaluate, real_distances
from blindspot.generators import claude_personas as cp
from blindspot.generators.baselines import gaussian_copula, independent_marginals, other_real_people
from blindspot.generators.claude_client import (CostLimitExceeded, ResponseCache, Usage, check_cost)

from .bakeoff1 import HEADER, _row, aggregate, cached_parsed, draw_candidates, eval_crowd, make_crowd, realism, select_per_persona
from .experiment0 import K, K_VALUES, N_BOOT, SEED, _assert_no_holdout, _git_commit, load_inputs
from .experiment1 import bunching_share, cost_of_all_responses
from .experiment_fp import restrict

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
MAX_RUN_USD = 8.0          # approved: standard API (the batch endpoint is blocked for this organisation)
MAX_TOTAL_USD = 12.0
SONNET, OPUS = "claude-sonnet-5-5", "claude-opus-5-5"
SELECT_LAMBDAS = (0.0, 0.5)
SELECT_SEEDS = 5
ASKED_SEEDS = 10
ASKED_B = 25
JOBS = (("sonnet", "gss"), ("opus", "gss"))     # new runs, GSS only (Twin skipped for now); haiku comes from the H3 cache


def model_names() -> dict[str, str]:
    return {"haiku": os.environ.get("CLAUDE_MODEL", "claude-haiku-5-5"), "sonnet": SONNET, "opus": OPUS}


# ---------- scoring helpers ----------
def single(target, crowd, rr, pool, seed: int = SEED) -> dict:
    m = evaluate(target, crowd, target.items, k=K, n_boot=N_BOOT, seed=seed, k_values=K_VALUES, real_dist=rr)
    r = {k: v for k, v in m.items() if not k.startswith("_") and k not in ("uncovered_ids", "per_family_entropy")}
    r["near_dup_share_vs_pool"] = near_duplicates(crowd, pool).share
    return {**aggregate([r]), "entropy": m["per_family_entropy"]["crowd"], "real_entropy": m["per_family_entropy"]["real"],
            "bunching_gt_80": bunching_share(crowd)}


def many(target, crowds: list, rr, pool, n_boot: int = 20) -> dict:
    return aggregate([eval_crowd(target, c, rr, pool, SEED + i, n_boot) for i, c in enumerate(crowds)])


def beats(row: dict, ref: dict) -> dict:
    """Point-estimate difference and whether the crowd's 95% CI lies entirely above the reference mean."""
    cov, ci = row["coverage"]["mean"], row["coverage_ci_first_seed"]
    r = ref["coverage"]["mean"]
    return {"diff": cov - r, "ci_above_reference": bool(ci[0] > r), "crowd": cov, "reference": r}


def references(target, pool, splits, rr) -> dict:
    demo = target.demographics
    ids = np.array(pool.answers.index)
    asked = [gaussian_copula(pool.subset(list(ids[np.random.default_rng(SEED + s).permutation(len(ids))[:ASKED_B]])),
                             demo, splits, SEED + s) for s in range(ASKED_SEEDS)]
    return {"shuffled_answers": single(target, independent_marginals(pool, demo, splits, SEED), rr, pool),
            "copula_on_pool": single(target, gaussian_copula(pool, demo, splits, SEED), rr, pool),
            "other_real_people": single(target, other_real_people(pool, len(target.answers), splits, SEED), rr, pool),
            f"copula_on_{ASKED_B}_asked (mean of {ASKED_SEEDS})": many(target, asked, rr, pool)}


def model_rows(target, pool, rr, mode, sampled, parsed, select: bool) -> dict:
    rows = {"claude_mode": single(target, mode, rr, pool), "claude_sampled": single(target, sampled, rr, pool)}
    if select:
        for lam in SELECT_LAMBDAS:
            crowds = []
            for s in range(SELECT_SEEDS):
                cands = draw_candidates(parsed, target.items, SEED + s)
                raw = select_per_persona(cands, [realism(c, pool) for c in cands], lam, target.items)
                crowds.append(make_crowd(raw, target.demographics, target.items, f"sel{lam}"))
            rows[f"select_from_50_lambda_{lam}"] = many(target, crowds, rr, pool, N_BOOT)
    return rows


# ---------- generation ----------
def make_plan(name: str, world: dict, model: str) -> cp.Plan:
    w = world[name]
    return cp.plan(name, w["full_target"].demographics, w["full_target"].items, model)


def estimate_runs(world: dict, client, cache, models: dict, batch: bool) -> dict:
    out = {}
    for key, ds in JOBS:
        plan = make_plan(ds, world, models[key])
        e = cp.estimate(plan, client, cache, batch)
        out[f"{key}:{ds}"] = e
        print(f"[estimate] {models[key]} on {ds}: {e['calls']} uncached calls, ~${e['cost_usd']:.2f}")
    return out


def run(out_dir: Path, estimate_only: bool, batch: bool = False) -> dict | None:
    from dotenv import load_dotenv

    started = time.perf_counter()
    load_dotenv(ROOT / ".env")
    models = model_names()
    inputs = {d.name: d for d in load_inputs()}                         # holdout already dropped
    world = {}
    inputs = {"gss": inputs["gss"]}                                     # Twin skipped: bigger models tested on GSS only
    for name, d in inputs.items():
        _assert_no_holdout(d.real.answers.index, d.splits, f"{name} real data")
        world[name] = {"splits": d.splits, "full_target": d.real.subset(d.splits["test"]),
                       "full_pool": d.real.subset(d.splits["pool"])}
    cache = ResponseCache()
    client = None
    if any(cache.get(s.key) is None for key, ds in JOBS for s in make_plan(ds, world, models[key]).specs):
        import anthropic

        client = anthropic.Anthropic()
    estimates = estimate_runs(world, client, cache, models, batch) if client else {}
    worst = max((e["cost_usd"] for e in estimates.values()), default=0.0)
    print(f"[estimate] most expensive single run ~${worst:.2f} (limit ${MAX_RUN_USD:.2f}); total ~${sum(e['cost_usd'] for e in estimates.values()):.2f}")
    for label, e in estimates.items():
        check_cost(e, MAX_RUN_USD)                                      # raises before anything is called
    if sum(e["cost_usd"] for e in estimates.values()) > MAX_TOTAL_USD:
        raise CostLimitExceeded(f"estimated total ${sum(e['cost_usd'] for e in estimates.values()):.2f} exceeds "
                                f"the ${MAX_TOTAL_USD:.2f} total limit; nothing was called")
    if estimate_only:
        return {"estimates": estimates}

    usage = {key: Usage(batch=batch) for key in models}
    crowds, parsed, plans = {}, {}, {}
    for key, ds in (("haiku", "gss"),) + JOBS:
        spent = sum(u.cost_usd for u in usage.values())
        if spent > MAX_TOTAL_USD:
            raise CostLimitExceeded(f"actual spend ${spent:.2f} exceeds the ${MAX_TOTAL_USD:.2f} total limit; stopping")
        plan = make_plan(ds, world, models[key])
        res = cp.execute(plan, client, cache, usage[key], seed=SEED, batch=batch, cost_limit=MAX_RUN_USD)
        if usage[key].cost_usd > MAX_RUN_USD:
            raise CostLimitExceeded(f"{models[key]} run cost ${usage[key].cost_usd:.2f}, over the ${MAX_RUN_USD:.2f} limit")
        crowds[(key, ds)], parsed[(key, ds)], plans[(key, ds)] = res, None, plan
        parsed[(key, ds)] = cached_parsed(plan, cache)

    report = {"settings": {"models": models, "k": K, "k_values": list(K_VALUES), "n_boot": N_BOOT, "seed": SEED,
                           "select_lambdas": list(SELECT_LAMBDAS), "select_seeds": SELECT_SEEDS, "asked_B": ASKED_B,
                           "asked_seeds": ASKED_SEEDS, "target_split": "test", "git_commit": _git_commit(),
                           "prompt_version": cp.PROMPT_VERSION, "batch": batch,
                           "limits_usd": {"per_run": MAX_RUN_USD, "total": MAX_TOTAL_USD},
                           "scope_note": "Bigger models (Sonnet 5.5, Opus 5.5) were tested on GSS only; Twin-2K-500 was not run."},
              "estimates": estimates, "datasets": {}}
    for ds, w in world.items():
        full_t, full_p = w["full_target"], w["full_pool"]
        sets = {"": (full_t, full_p)}
        if ds == "gss":
            live = [i.code for i in full_t.items if i.live_ok]
            sets = {"24 items": (full_t, full_p), "20 live items": (restrict(full_t, live), restrict(full_p, live))}
        else:
            sets = {"30 items": (full_t, full_p)}
        block = {}
        for label, (t, p) in sets.items():
            rr = real_distances(t, t.items)
            codes = [i.code for i in t.items]
            entry = {"n_target": len(t.answers), "n_pool": len(p.answers), "n_items": len(codes),
                     "references": references(t, p, w["splits"], rr), "models": {}}
            for key in models:
                if (key, ds) not in crowds:
                    continue
                res = crowds[(key, ds)]
                mode, sampled = restrict(res.mode, codes), restrict(res.sampled, codes)
                entry["models"][key] = model_rows(t, p, rr, mode, sampled, parsed[(key, ds)], select=(ds == "gss" and label == "20 live items"))
                entry["models"][key]["_failures"] = {"people_with_missing_items": len(res.failures)}
            ref_s = entry["references"]["shuffled_answers"]
            ref_c = entry["references"][f"copula_on_{ASKED_B}_asked (mean of {ASKED_SEEDS})"]
            entry["paired"] = {key: {name: {"vs_shuffled": beats(row, ref_s), f"vs_copula_on_{ASKED_B}": beats(row, ref_c)}
                                     for name, row in rows.items() if not name.startswith("_")}
                               for key, rows in entry["models"].items()}
            block[label] = entry
        report["datasets"][ds] = block
    report["cost"] = {key: {"actual_this_run": usage[key].to_dict(),
                            "all_responses_used": cost_of_all_responses({d: plans[(key, d)] for (k2, d) in plans if k2 == key}, cache, batch)}
                      for key in models}
    report["runtime_seconds"] = round(time.perf_counter() - started, 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "experiment1b.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (out_dir / "experiment1b.md").write_text(to_markdown(report), encoding="utf-8")
    return report


# ---------- markdown ----------
def _entropy(entry: dict) -> list[str]:
    cols = [(f"{k}:{n}", r) for k, rows in entry["models"].items() for n, r in rows.items() if "entropy" in r]
    lines = ["| family | real | " + " | ".join(c for c, _ in cols) + " |", "| --- |" + " --- |" * (len(cols) + 1)]
    real = next(iter(entry["references"].values()))["real_entropy"]
    for fam, h in real.items():
        lines.append(f"| {fam} | {h:.2f} | " + " | ".join(f"{r['entropy'][fam]:.2f}" for _, r in cols) + " |")
    lines += ["", "Items with >80% of people on one option: " + ", ".join(f"{c} {r['bunching_gt_80']:.2f}" for c, r in cols)]
    return lines


def to_markdown(r: dict) -> str:
    s, cost = r["settings"], r["cost"]
    names = {k: v for k, v in s["models"].items()}
    lines = ["# Experiment 1b: does a bigger model reduce the coverage gap? (test split only)", "",
             f"Models: {', '.join(f'{k} = `{v}`' for k, v in names.items())}. Same H3 generator and prompts "
             f"({s['prompt_version']}, demographics only, API-default sampling), {'batch' if s['batch'] else 'standard'} API with "
             f"bounded concurrency. **{s['scope_note']}** Target = test split, holdout never "
             f"used. k = {s['k']}, {s['n_boot']} bootstrap resamples; multi-seed rows show the mean (seed range) with the "
             f"first seed's CI. Commit `{s['git_commit']}`. Runtime {r['runtime_seconds']} s.", "",
             "Cost this run: " + "; ".join(
                 f"{k} ${c['actual_this_run']['cost_usd']:.2f} ({c['actual_this_run']['calls']} calls; all responses used "
                 f"${c['all_responses_used']['cost_usd']:.2f})" for k, c in cost.items()) + ".", ""]
    for ds, block in r["datasets"].items():
        for label, entry in block.items():
            lines += [f"## {ds}, {label} (target n = {entry['n_target']}, pool n = {entry['n_pool']})", ""] + HEADER
            for key, rows in entry["models"].items():
                lines += [_row(f"{key}: {n}", a) for n, a in rows.items() if not n.startswith("_")]
            lines += [_row(f"ref: {n}", a) for n, a in entry["references"].items()]
            lines += ["", "Per-family normalized entropy and bunching:", ""] + _entropy(entry)
            lines += ["", "Paired note (does the crowd beat the references?):", ""]
            for key, rows in entry["paired"].items():
                for name, p in rows.items():
                    a, b = p["vs_shuffled"], p[f"vs_copula_on_{s['asked_B']}"]
                    lines.append(f"- {key} {name}: {a['crowd']:.3f}; vs shuffled answers {a['diff']:+.3f} "
                                 f"({'CI above' if a['ci_above_reference'] else 'not above'}), vs copula on {s['asked_B']} asked "
                                 f"{b['diff']:+.3f} ({'CI above' if b['ci_above_reference'] else 'not above'})")
            fails = {k: rows.get("_failures", {}).get("people_with_missing_items") for k, rows in entry["models"].items()}
            lines += ["", f"People with a missing item after retry: {fails}", ""]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 1b: bigger models, test split only")
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=RESULTS)
    args = parser.parse_args()
    try:
        out = run(args.out_dir, args.estimate_only)
    except CostLimitExceeded as e:
        raise SystemExit(f"STOPPED by cost guard: {e}") from None
    if args.estimate_only:
        print(json.dumps({k: v["cost_usd"] for k, v in out["estimates"].items()}, indent=1))
        return
    print((args.out_dir / "experiment1b.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
