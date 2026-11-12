"""Experiment 2 (H4): does asking for the right real people and learning archetypes beat the baselines?

`python -m blindspot.eval.experiment2 [--estimate-only] [--max-cost 5] [--allow-worst-case]`
Target = TEST split; the agent asks only POOL people; HOLDOUT is dropped at load and never used.
At each budget of real people asked (25, 50, 75, 100) we compare:
  blindspot_targeted : the full loop            blindspot_random : same loop, random asking (ablation)
  copula_on_asked    : Gaussian copula fit only on the people Blindspot (targeted) asked at that budget
  claude_plus_asked  : claude_mode crowd with those asked real people swapped in for synthetic ones
plus fixed references from experiment1.json (claude_mode = budget 0, claude_sampled, copula on the
full pool, other_real_people, retest).
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from blindspot.agent.budget import Budget
from blindspot.agent.library import RECIPE_FRAME
from blindspot.agent.loop import LoopConfig, LoopResult, _public, run_loop
from blindspot.coverage import evaluate, real_distances
from blindspot.data.schema import Respondents
from blindspot.generators import claude_personas as cp
from blindspot.generators.baselines import gaussian_copula
from blindspot.generators.claude_client import (DEFAULT_COST_LIMIT_USD, CostLimitExceeded, ResponseCache, Usage,
                                                estimate_cost)
from blindspot.space import map_coords

from .experiment0 import DatasetInput, _assert_no_holdout, _git_commit, load_inputs

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
SEEDS = (42, 43, 44)
ROUNDS = 4
PER_ROUND = 25
N_BOOT = 50
AGENT_CALLS_PER_ROUND = 6            # up to 5 cluster descriptions + 1 seek (worst case)
AGENT_CALL_COST_USD = 0.0015         # generous per-call allowance at standard prices
ASSUMED_ARCHETYPE_SHARE = 0.5        # only for the printed "expected" estimate
REFERENCE_ORDER = ("claude_mode", "claude_sampled", "gaussian_copula", "other_real_people", "retest")
METHODS = ("blindspot_targeted", "blindspot_random", "copula_on_asked", "claude_plus_asked")


# ---------------- estimate ----------------
def crowds_per_dataset(seeds, rounds: int) -> int:
    """Persona-crowd generations per dataset: targeted has `rounds`; random shares round 1 with targeted."""
    return len(seeds) * (rounds + (rounds - 1))


def estimate_run(inputs: list[DatasetInput], client, model: str, seeds, rounds: int, max_share: float,
                 batch: bool) -> dict:
    """Worst case (every archetype-assigned person is a fresh call every round) and an expected case."""
    per_dataset, worst, expected = {}, 0.0, 0.0
    for d in inputs:
        target = d.real.subset(d.splits["test"])
        dummy = RECIPE_FRAME + "pattern: " + " ".join(["word"] * 120)
        sample = cp.plan(d.name, target.demographics.iloc[:1], target.items, model, recipes=[dummy]).specs[0].params
        tokens = client.messages.count_tokens(model=model, system=sample["system"],
                                              messages=sample["messages"]).input_tokens
        tokens += len(json.dumps(sample["output_config"])) // 3
        n_calls = int(len(target.answers) * crowds_per_dataset(seeds, rounds))
        out_tokens = cp._output_tokens_estimate(target.items)
        w = estimate_cost(int(n_calls * max_share), tokens, out_tokens, model, batch)["cost_usd"]
        e = estimate_cost(int(n_calls * min(max_share, ASSUMED_ARCHETYPE_SHARE)), tokens, out_tokens, model,
                          batch)["cost_usd"]
        agent = len(seeds) * 2 * rounds * AGENT_CALLS_PER_ROUND * AGENT_CALL_COST_USD
        per_dataset[d.name] = {"persona_calls_worst": int(n_calls * max_share), "tokens_per_call": tokens,
                               "worst_usd": round(w + agent, 2), "expected_usd": round(e + agent / 2, 2)}
        worst += w + agent
        expected += e + agent / 2
    return {"per_dataset": per_dataset, "worst_case_usd": round(worst, 2), "expected_usd": round(expected, 2),
            "assumed_archetype_share_for_expected": min(max_share, ASSUMED_ARCHETYPE_SHARE),
            "max_archetype_share": max_share, "batch": batch}


# ---------------- one dataset ----------------
def swap_in_asked(baseline: Respondents, asked: Respondents) -> Respondents:
    """claude_mode crowd with len(asked) synthetic people replaced by the asked real people."""
    keep = baseline.subset(list(baseline.answers.index[len(asked.answers):]))
    demo = pd.concat([asked.demographics[keep.demographics.columns], keep.demographics])
    return Respondents(pd.concat([asked.answers, keep.answers]), demo, baseline.items, "baseline:claude_plus_asked")


def _eval(target: Respondents, crowd: Respondents, rr: np.ndarray, seed: int, n_boot: int) -> dict:
    return _public(evaluate(target, crowd, target.items, k=5, n_boot=n_boot, seed=seed, k_values=(5,), real_dist=rr))


def run_seed(d: DatasetInput, seed: int, ctx: dict) -> dict:
    """Both variants for one seed (sequential: they share round 1) plus the budget-matched baselines."""
    target, pool = ctx["target"], ctx["pool"]
    runs = {}
    for variant in ("targeted", "random"):
        cfg = LoopConfig(d.name, seed, variant, ctx["rounds"], ctx["per_round"],
                         max_archetype_share=ctx["max_share"], n_boot=ctx["n_boot"], batch=ctx["batch"],
                         out_dir=ctx["out_dir"] / "agent" / d.name / f"{variant}_seed{seed}")
        runs[variant] = run_loop(cfg, d.real, d.splits, ctx["baseline"], ctx["rr"], ctx["client"], ctx["cache"],
                                 ctx["model"], ctx["budget"], log=ctx["log"])
    baselines = {"copula_on_asked": [], "claude_plus_asked": []}
    for asked_ids in runs["targeted"].asked_by_round:
        asked = pool.subset(asked_ids)
        copula = gaussian_copula(asked, target.demographics, d.splits, seed)
        baselines["copula_on_asked"].append(_eval(target, copula, ctx["rr"], seed, ctx["n_boot"]))
        baselines["claude_plus_asked"].append(
            _eval(target, swap_in_asked(ctx["baseline"], asked), ctx["rr"], seed, ctx["n_boot"]))
    return {"runs": runs, "baselines": baselines}


def prepare(d: DatasetInput, client, cache, model: str, budget: Budget, batch: bool, out_dir: Path, log) -> dict:
    """Test target, pool, the budget-0 claude_mode crowd (cached from H3) and the real-real distances."""
    target, pool = d.real.subset(d.splits["test"]), d.real.subset(d.splits["pool"])
    usage = Usage(batch=batch)
    budget.register(usage)
    plan = cp.plan(d.name, target.demographics, target.items, model)
    budget.check(cp.estimate(plan, client, cache, batch)["cost_usd"], f"{d.name} baseline crowd")
    baseline = cp.execute(plan, client, cache, usage, seed=SEEDS[0], batch=batch, log=log).mode
    return {"target": target, "pool": pool, "baseline": baseline, "rr": real_distances(target, target.items),
            "client": client, "cache": cache, "model": model, "budget": budget, "batch": batch,
            "out_dir": out_dir, "log": log}


def sum_usage(per_seed: dict, batch: bool) -> Usage:
    total = Usage(batch=batch)
    for res in per_seed.values():
        for run in res["runs"].values():
            total.calls += run.usage.calls
            total.cost_usd += run.usage.cost_usd
            total.errors += run.usage.errors
    return total


def run_dataset(d: DatasetInput, client, cache, model: str, budget: Budget, seeds, rounds: int, per_round: int,
                max_share: float, batch: bool, workers: int, n_boot: int, out_dir: Path, maps: bool, log) -> dict:
    started = time.perf_counter()
    _assert_no_holdout(d.real.answers.index, d.splits, f"{d.name} real data")
    ctx = prepare(d, client, cache, model, budget, batch, out_dir, log)
    ctx.update(rounds=rounds, per_round=per_round, max_share=max_share, n_boot=n_boot)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        per_seed = dict(zip(seeds, pool.map(lambda s: run_seed(d, s, ctx), seeds)))
    for res in per_seed.values():                       # no holdout id may appear in any asked list
        for run in res["runs"].values():
            for rd in run.rounds:
                _assert_no_holdout(rd["asked_new"], d.splits, f"{d.name} asked ids")
    map_files = write_maps(d, ctx, per_seed[seeds[0]]["runs"]["targeted"], seeds[0], out_dir) if maps else {}
    agent_path = out_dir / "agent" / d.name / "rounds.json"
    agent_path.parent.mkdir(parents=True, exist_ok=True)
    agent_path.write_text(json.dumps({
        "dataset": d.name, "per_round": per_round, "rounds": rounds,
        "runs": {v: {str(s): per_seed[s]["runs"][v].rounds for s in seeds} for v in ("targeted", "random")}},
        indent=1), encoding="utf-8")
    return {"title": d.title, "n_target": len(ctx["target"].answers), "n_pool": len(ctx["pool"].answers),
            "budget_0_claude_mode": _eval(ctx["target"], ctx["baseline"], ctx["rr"], seeds[0], n_boot),
            "per_seed": summarize_seeds(per_seed), "map_files": map_files,
            "agent_log": f"agent/{d.name}/rounds.json", "usage": sum_usage(per_seed, batch).to_dict(),
            "runtime_seconds": round(time.perf_counter() - started, 1)}


def summarize_seeds(per_seed: dict) -> dict:
    """{seed: {method: [metrics per budget], diagnostics}} JSON-ready."""
    out = {}
    for seed, res in per_seed.items():
        runs = res["runs"]
        out[str(seed)] = {
            "blindspot_targeted": [r["metrics"] for r in runs["targeted"].rounds],
            "blindspot_random": [r["metrics"] for r in runs["random"].rounds],
            **res["baselines"],
            "diagnostics": {v: [{"round": r["round"], "n_archetypes": len(r["library"]["archetypes"]),
                                 "archetype_weight": round(1 - r["library"]["no_archetype_weight"], 3),
                                 "near_duplicate_share": r["near_duplicates"]["share"],
                                 "n_uncovered_asked": len(r["uncovered_asked"]),
                                 "cost_usd_cumulative": r["cost_usd_cumulative"]}
                                for r in runs[v].rounds] for v in ("targeted", "random")}}
    return out


def write_maps(d: DatasetInput, ctx: dict, run: LoopResult, seed: int, out_dir: Path) -> dict:
    """GSS (live demo) maps: test vs the crowd at each round, with each real person's covered flag."""
    if d.name != "gss":
        return {}
    target = ctx["target"]
    base = evaluate(target, ctx["baseline"], target.items, k=5, n_boot=10, seed=seed, k_values=(5,),
                    real_dist=ctx["rr"])
    crowds = {"agent_round0": ctx["baseline"], **{f"agent_round{r}": c for r, c in enumerate(run.crowds, start=1)}}
    flags = {"agent_round0": {"real_covered": base["_real_covered"], "synthetic_inside": base["_synthetic_inside"]},
             **{f"agent_round{r}": e for r, e in enumerate(run.private_eval, start=1)}}
    paths = map_coords(target, crowds, seed=seed, covered_by_name=flags, dataset=d.name, out_dir=out_dir / "maps")
    return {n: f"maps/{p.name}" for n, p in paths.items()}


# ---------------- aggregate + markdown ----------------
def _cell(values: list[float]) -> str:
    return f"{np.mean(values):.3f} ({min(values):.3f}-{max(values):.3f})"


def aggregate(per_seed: dict, metric: str) -> dict[str, list[list[float]]]:
    """method -> per budget -> values across seeds."""
    seeds = list(per_seed)
    return {m: [[per_seed[s][m][i][metric] for s in seeds] for i in range(len(per_seed[seeds[0]][m]))]
            for m in METHODS}


def dataset_markdown(ds: dict, refs: dict, per_round: int, n_seeds: int) -> list[str]:
    n_budgets = len(next(iter(ds["per_seed"].values()))["blindspot_targeted"])
    budgets = [per_round * (i + 1) for i in range(n_budgets)]
    lines = [f"## {ds['title']} (target n = {ds['n_target']}, pool n = {ds['n_pool']}, {n_seeds} seeds)", ""]
    for metric in ("coverage", "precision"):
        lines += [f"**{metric}**, mean (min-max) over seeds, by number of real people asked:", "",
                  "| method | " + " | ".join(str(b) for b in budgets) + " |", "| --- |" + " --- |" * len(budgets)]
        lines += [f"| {m} | " + " | ".join(_cell(v) for v in vals) + " |"
                  for m, vals in aggregate(ds["per_seed"], metric).items()] + [""]
    base = ds["budget_0_claude_mode"]
    lines += ["Fixed references (test split, k = 5):", "", "| reference | coverage | precision |", "| --- | --- | --- |",
              f"| claude_mode, budget 0 | {base['coverage']:.3f} | {base['precision']:.3f} |"]
    lines += [f"| {n} | {m['coverage']:.3f} | {m['precision']:.3f} |" for n, m in refs.items()]
    seeds = list(ds["per_seed"])
    diag = ds["per_seed"][seeds[0]]["diagnostics"]["targeted"]
    worst_dup = max(r["near_duplicate_share"] for s in seeds for v in ("targeted", "random")
                    for r in ds["per_seed"][s]["diagnostics"][v])
    lines += ["", f"Near-duplicate share (synthetic people within 0.02 of an asked real person), worst over all "
                  f"runs and rounds: {worst_dup:.2%} (limit 5%). Seed {seeds[0]} targeted archetypes per round: "
                  f"{[r['n_archetypes'] for r in diag]}.",
              f"Spend for this dataset: ${ds['usage']['cost_usd']:.2f} over {ds['usage']['calls']} billed calls "
              f"({len(ds['usage']['errors'])} errors). Runtime {ds['runtime_seconds']} s.", ""]
    return lines


def to_markdown(report: dict) -> str:
    s = report["settings"]
    lines = ["# Experiment 2: the self-improving loop vs baselines at equal numbers of real people asked", "",
             f"Model `{s['model']}`, seeds {s['seeds']}, {s['rounds']} rounds of {s['per_round']} people, "
             f"max archetype share {s['max_archetype_share']}, k = 5, {s['n_boot']} bootstrap resamples per point. "
             f"Target = test split, agent asks pool only, holdout untouched. Commit `{s['git_commit']}`.", "",
             f"Cost: worst-case estimate ${report['estimate']['worst_case_usd']:.2f}, expected "
             f"${report['estimate']['expected_usd']:.2f}; actual ${report['cost']['actual_usd']:.2f} "
             f"(limit ${s['cost_limit_usd']:.2f}). Runtime {report['runtime_seconds']} s.", ""]
    for name, ds in report["datasets"].items():
        lines += dataset_markdown(ds, report["references"][name], s["per_round"], len(s["seeds"]))
    return "\n".join(lines)


# ---------------- run ----------------
def load_references(path: Path, names: list[str]) -> dict:
    exp1 = json.loads(path.read_text(encoding="utf-8"))["datasets"]
    return {n: {c: exp1[n]["crowds"][c] for c in REFERENCE_ORDER if c in exp1[n]["crowds"]} for n in names}


def run(inputs: list[DatasetInput], client, out_dir: Path = RESULTS, cache: ResponseCache | None = None,
        model: str | None = None, cost_limit: float = DEFAULT_COST_LIMIT_USD, seeds=SEEDS, rounds: int = ROUNDS,
        per_round: int = PER_ROUND, max_share: float = 1.0, batch: bool = True, workers: int = 3,
        n_boot: int = N_BOOT, maps: bool = True, allow_worst_case: bool = False, estimate_only: bool = False,
        exp1_path: Path = RESULTS / "experiment1.json", log=print) -> dict:
    started = time.perf_counter()
    cache, model = cache or ResponseCache(), model or os.environ["CLAUDE_MODEL"]
    estimate = estimate_run(inputs, client, model, seeds, rounds, max_share, batch)
    print(f"[estimate] worst case ${estimate['worst_case_usd']:.2f}, expected ${estimate['expected_usd']:.2f} "
          f"(assuming {estimate['assumed_archetype_share_for_expected']:.0%} of synthetic people get an "
          f"archetype); limit ${cost_limit:.2f}")
    for name, e in estimate["per_dataset"].items():
        print(f"[estimate]   {name}: {e['persona_calls_worst']} persona calls worst case, ~{e['tokens_per_call']} "
              f"input tokens each, worst ${e['worst_usd']:.2f}, expected ${e['expected_usd']:.2f}")
    if estimate_only:
        return {"estimate": estimate}
    if estimate["worst_case_usd"] > cost_limit and not allow_worst_case:
        raise CostLimitExceeded(
            f"worst-case estimate ${estimate['worst_case_usd']:.2f} exceeds the ${cost_limit:.2f} limit. Nothing "
            f"was called. Options: --allow-worst-case (spend is still stopped per round at the limit), "
            f"--max-archetype-share to cap the worst case, fewer --seeds, or a higher --max-cost.")
    budget = Budget(cost_limit)
    datasets = {d.name: run_dataset(d, client, cache, model, budget, seeds, rounds, per_round, max_share, batch,
                                    workers, n_boot, out_dir, maps, log) for d in inputs}
    report = {"settings": {"model": model, "seeds": list(seeds), "rounds": rounds, "per_round": per_round,
                           "max_archetype_share": max_share, "n_boot": n_boot, "batch": batch,
                           "cost_limit_usd": cost_limit, "git_commit": _git_commit()},
              "estimate": estimate, "cost": {"actual_usd": round(budget.spent, 4)}, "datasets": datasets,
              "references": load_references(exp1_path, [d.name for d in inputs]),
              "runtime_seconds": round(time.perf_counter() - started, 1)}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "experiment2.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (out_dir / "experiment2.md").write_text(to_markdown(report), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 2: the self-improving loop")
    parser.add_argument("--max-cost", type=float, default=DEFAULT_COST_LIMIT_USD)
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--allow-worst-case", action="store_true",
                        help="run even if the worst-case estimate is over the limit (spend is still capped per round)")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--datasets", nargs="+", default=["gss", "twin2k"], choices=["gss", "twin2k"])
    parser.add_argument("--max-archetype-share", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--no-batch", action="store_true")
    args = parser.parse_args()

    import anthropic
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    inputs = sorted((d for d in load_inputs() if d.name in args.datasets), key=lambda d: args.datasets.index(d.name))
    try:
        run(inputs, anthropic.Anthropic(), cost_limit=args.max_cost, seeds=tuple(args.seeds),
            max_share=args.max_archetype_share, batch=not args.no_batch, workers=args.workers,
            allow_worst_case=args.allow_worst_case, estimate_only=args.estimate_only)
    except CostLimitExceeded as e:
        raise SystemExit(f"STOPPED by cost guard: {e}") from None
    if not args.estimate_only:
        print((RESULTS / "experiment2.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
