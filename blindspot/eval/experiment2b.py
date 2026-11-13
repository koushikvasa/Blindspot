"""Experiment 2b (H4b): the redesigned loop (detect, propose, verify, escalate, assemble) on GSS.

`python -m blindspot.eval.experiment2b [--max-cost 5] [--seeds 42 43 44]`
Compared at budgets of 25/50/75/100 real people asked (test split is the target; the agent asks pool
people only; holdout is dropped at load and never used):
  h4b_verify, h4b_ablation (step 3 skipped: every recipe accepted), copula_on_asked, claude_plus_asked
and, for reference, the earlier H4 loop (experiment2.json) and the fixed references of experiment1.json.
claude_plus_asked swaps asked real people INTO the crowd (the rule-2 baseline); the H4b crowds never do.
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from blindspot.agent.budget import Budget
from blindspot.agent.loop2 import LoopConfigB, run_loop_b
from blindspot.generators import claude_personas as cp
from blindspot.generators.baselines import gaussian_copula
from blindspot.generators.claude_client import (DEFAULT_COST_LIMIT_USD, CostLimitExceeded, ResponseCache, Usage,
                                                estimate_cost)

from .experiment0 import DatasetInput, _assert_no_holdout, _git_commit, load_inputs
from .experiment2 import REFERENCE_ORDER, _cell, _eval, prepare, swap_in_asked

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
SEEDS = (42, 43, 44)
ROUNDS = 4
PER_ROUND = 25
N_BOOT = 50
METHODS = ("h4b_verify", "h4b_ablation", "copula_on_asked", "claude_plus_asked")
ESTIMATE_CLUSTERS = 5
ESTIMATE_ATTEMPTS = 2


def estimate_worst_case(d: DatasetInput, client, model: str, seeds, rounds: int) -> dict:
    """Printed before the run: every round at the cluster/attempt maximum, nothing cached. Spend is also
    hard-stopped at the limit before every step."""
    target = d.real.subset(d.splits["test"])
    plan = cp.plan(d.name, target.demographics.iloc[:1], target.items, model, recipes=["pattern"])
    params = plan.specs[0].params
    tokens = client.messages.count_tokens(model=model, system=params["system"],
                                          messages=params["messages"]).input_tokens
    verify_calls = ESTIMATE_CLUSTERS * ESTIMATE_ATTEMPTS * 10 * (1 + 3)     # control + 3 variants, 10 people each
    n_target = len(target.answers)
    n_calls = len(seeds) * rounds * (verify_calls + n_target + n_target)    # verify loop + ablation loop
    cost = estimate_cost(n_calls, tokens + 400, cp._output_tokens_estimate(target.items), model, batch=False)
    return {"worst_case_calls": n_calls, "tokens_per_call": tokens, "worst_case_usd": cost["cost_usd"]}


def run_seed(d: DatasetInput, seed: int, ctx: dict) -> dict:
    target, pool = ctx["target"], ctx["pool"]
    runs = {}
    for variant in ("verify", "ablation"):
        cfg = LoopConfigB(d.name, seed, variant, ctx["rounds"], ctx["per_round"], n_boot=ctx["n_boot"],
                          out_dir=ctx["out_dir"] / "agent" / f"{d.name}_h4b" / f"{variant}_seed{seed}")
        runs[variant] = run_loop_b(cfg, d.real, d.splits, ctx["baseline"], ctx["rr"], ctx["client"], ctx["cache"],
                                   ctx["model"], ctx["budget"], log=ctx["log"])
    baselines = {"copula_on_asked": [], "claude_plus_asked": []}
    for asked_ids in runs["verify"].asked_by_round:
        asked = pool.subset(asked_ids)
        copula = gaussian_copula(asked, target.demographics, d.splits, seed)
        baselines["copula_on_asked"].append(_eval(target, copula, ctx["rr"], seed, ctx["n_boot"]))
        baselines["claude_plus_asked"].append(
            _eval(target, swap_in_asked(ctx["baseline"], asked), ctx["rr"], seed, ctx["n_boot"]))
    return {"runs": runs, "baselines": baselines}


def _recipe_row(r: dict) -> dict:
    return {"round": r["round"], "n_clusters": len(r["clusters"]),
            "n_merged": sum(c["action"] == "merged" for c in r["clusters"]),
            "n_proposed": r["n_variants_proposed"], "n_accepted": r["n_variants_accepted"],
            "n_rejected": r["n_variants_proposed"] - r["n_variants_accepted"],
            "n_cannot_imagine": r["n_clusters_cannot_imagine"], "n_replaced": r["assembly"]["n_replaced"],
            "dropped_by_guard": sum(len(a["dropped_by_guard"]) for c in r["clusters"] for a in c.get("attempts", []))}


def summarize(per_seed: dict) -> dict:
    out = {}
    for seed, res in per_seed.items():
        runs = res["runs"]
        out[str(seed)] = {
            "h4b_verify": [r["metrics"] for r in runs["verify"].rounds],
            "h4b_ablation": [r["metrics"] for r in runs["ablation"].rounds], **res["baselines"],
            "recipes": {v: [_recipe_row(r) for r in runs[v].rounds] for v in ("verify", "ablation")},
            "detection": [r["detection"] for r in runs["verify"].rounds],
            "escalation": [{"round": r["round"], "asking_mode": r["asking_mode"],
                            "escalation_used_for_this_round": r["escalation_used_for_this_round"],
                            "asking_report": r["asking_report"], "escalation_decided_after_round": r["escalation"]}
                           for r in runs["verify"].rounds],
            "near_duplicate_share": {v: [r["near_duplicates"]["share"] for r in runs[v].rounds]
                                     for v in ("verify", "ablation")}}
    return out


def run_dataset(d: DatasetInput, client, cache, model: str, budget: Budget, seeds, rounds: int, per_round: int,
                workers: int, n_boot: int, out_dir: Path, log) -> dict:
    started = time.perf_counter()
    _assert_no_holdout(d.real.answers.index, d.splits, f"{d.name} real data")
    ctx = prepare(d, client, cache, model, budget, False, out_dir, log)
    ctx.update(rounds=rounds, per_round=per_round, n_boot=n_boot)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        per_seed = dict(zip(seeds, pool.map(lambda s: run_seed(d, s, ctx), seeds)))
    usage = Usage(batch=False)
    for res in per_seed.values():
        for run in res["runs"].values():
            for r in run.rounds:
                _assert_no_holdout(r["asked_new"], d.splits, f"{d.name} asked ids")
            usage.calls += run.usage.calls
            usage.cost_usd += run.usage.cost_usd
            usage.errors += run.usage.errors
    path = out_dir / "agent" / f"{d.name}_h4b" / "rounds.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"dataset": d.name, "per_round": per_round, "rounds": rounds, "runs": {
        v: {str(s): per_seed[s]["runs"][v].rounds for s in seeds} for v in ("verify", "ablation")}}, indent=1),
        encoding="utf-8")
    return {"title": d.title, "n_target": len(ctx["target"].answers), "n_pool": len(ctx["pool"].answers),
            "budget_0_claude_mode": _eval(ctx["target"], ctx["baseline"], ctx["rr"], seeds[0], n_boot),
            "per_seed": summarize(per_seed), "usage": usage.to_dict(),
            "agent_log": f"agent/{d.name}_h4b/rounds.json", "runtime_seconds": round(time.perf_counter() - started, 1)}


# ---------------- markdown ----------------
def aggregate(per_seed: dict, metric: str) -> dict:
    seeds = list(per_seed)
    return {m: [[per_seed[s][m][i][metric] for s in seeds] for i in range(len(per_seed[seeds[0]][m]))]
            for m in METHODS}


def _old_h4_rows(exp2_path: Path, seeds: list[str], metric: str) -> list[list[float]] | None:
    if not exp2_path.exists():
        return None
    ds = json.loads(exp2_path.read_text(encoding="utf-8"))["datasets"].get("gss")
    if not ds or not all(s in ds["per_seed"] for s in seeds):
        return None
    n = len(ds["per_seed"][seeds[0]]["blindspot_targeted"])
    return [[ds["per_seed"][s]["blindspot_targeted"][i][metric] for s in seeds] for i in range(n)]


def _recipe_table(ds: dict, seeds: list[str]) -> list[str]:
    lines = ["| seed | variant | clusters | merged | proposed | accepted | rejected | dropped by guard | "
             "cannot imagine | people replaced (last round) |", "| --- |" + " --- |" * 9]
    keys = ("n_clusters", "n_merged", "n_proposed", "n_accepted", "n_rejected", "dropped_by_guard", "n_cannot_imagine")
    for seed in seeds:
        for v in ("verify", "ablation"):
            rows = ds["per_seed"][seed]["recipes"][v]
            t = {k: sum(r[k] for r in rows) for k in keys}
            lines.append(f"| {seed} | {v} | " + " | ".join(str(t[k]) for k in keys) + f" | {rows[-1]['n_replaced']} |")
    return lines


def _detection_table(ds: dict, seeds: list[str]) -> list[str]:
    lines = ["| seed | round | asked | old rule (k-NN ball) | new rule (95th pct NN) | new-rule threshold |",
             "| --- | --- | --- | --- | --- | --- |"]
    for seed in seeds:
        for r, det in enumerate(ds["per_seed"][seed]["detection"], start=1):
            lines.append(f"| {seed} | {r} | {det['n_asked']} | {det['n_uncovered_old_rule']} | "
                         f"{det['n_uncovered_new_rule']} | {det['threshold']} |")
    return lines


def _escalation_log(ds: dict, seed: str) -> list[str]:
    lines = []
    for e in ds["per_seed"][seed]["escalation"]:
        used, after = e["escalation_used_for_this_round"], e["escalation_decided_after_round"]
        lines.append(f"- **Round {e['round']}** asked {e['asking_mode']}: {used['next_round']}")
        if used.get("top_segments"):
            seen = "; ".join(f"{t['field']}={t['value']} {t['n_uncovered']}/{t['n']} ({t['rate']:.0%})"
                             for t in used["top_segments"][:5])
            lines.append(f"  - segments seen when deciding: {seen} ({used['segments_below_min_n']} segments had "
                         f"fewer than {used['min_segment_n']} people)")
        if e["asking_report"]:
            drawn = "; ".join(f"{a.get('label', 'fallback to random')} got {a.get('found', a.get('fallback_random'))}"
                              f" of {a.get('requested', a.get('fallback_random'))}" for a in e["asking_report"])
            lines.append(f"  - drawn: {drawn}")
        stuck = f"; cannot imagine: {after['cannot_imagine_clusters']}" if after.get("triggered") else ""
        lines.append(f"  - decided after this round: {after['next_round']}{stuck}")
    return lines


def to_markdown(report: dict) -> str:
    s, ds = report["settings"], report["datasets"]["gss"]
    seeds = list(ds["per_seed"])
    budgets = [s["per_round"] * (i + 1) for i in range(s["rounds"])]
    lines = ["# Experiment 2b: the redesigned loop (detect, propose, verify, escalate, assemble) on GSS", "",
             f"Model `{s['model']}`, seeds {s['seeds']}, {s['rounds']} rounds of {s['per_round']} people, k = 5. "
             f"Target = test split (n = {ds['n_target']}); agent asks pool only; holdout untouched. "
             f"Commit `{s['git_commit']}`. Cost: actual ${report['cost']['actual_usd']:.2f} (hard stop "
             f"${s['cost_limit_usd']:.2f}; worst-case estimate ${report['estimate']['worst_case_usd']:.2f}). "
             f"The H4b crowds contain NO real respondents' rows; claude_plus_asked does (the rule-2 baseline).", ""]
    for metric in ("coverage", "precision"):
        lines += [f"**{metric}**, mean (min-max) over seeds, by number of real people asked:", "",
                  "| method | " + " | ".join(map(str, budgets)) + " |", "| --- |" + " --- |" * len(budgets)]
        lines += [f"| {m} | " + " | ".join(_cell(v) for v in vals) + " |"
                  for m, vals in aggregate(ds["per_seed"], metric).items()]
        old = _old_h4_rows(RESULTS / "experiment2.json", seeds, metric)
        if old:
            lines.append("| (earlier H4 loop, for reference) | " + " | ".join(_cell(v) for v in old) + " |")
        lines.append("")
    base = ds["budget_0_claude_mode"]
    lines += ["Fixed references (test split):", "", "| reference | coverage | precision |", "| --- | --- | --- |",
              f"| claude_mode, budget 0 | {base['coverage']:.3f} | {base['precision']:.3f} |"]
    lines += [f"| {n} | {m['coverage']:.3f} | {m['precision']:.3f} |" for n, m in report["references"].items()]
    lines += ["", "## Recipe accept/reject counts (summed over rounds 1-4)", ""] + _recipe_table(ds, seeds)
    lines += ["", "## Uncovered asked people per round: old rule vs new rule (verify loop)", ""]
    lines += _detection_table(ds, seeds)
    lines += ["", f"## Escalation log, seed {seeds[0]} (verify loop)", ""] + _escalation_log(ds, seeds[0])
    worst = max(x for sd in ds["per_seed"].values() for v in sd["near_duplicate_share"].values() for x in v)
    lines += ["", f"Near-duplicate share, worst over all runs and rounds: {worst:.2%} (limit 5%). "
                  f"Billed calls: {ds['usage']['calls']}, errors: {len(ds['usage']['errors'])}. "
                  f"Wall-clock (includes any laptop sleep): {ds['runtime_seconds']} s.", ""]
    return "\n".join(lines)


def run(inputs: list[DatasetInput], client, out_dir: Path = RESULTS, cache: ResponseCache | None = None,
        model: str | None = None, cost_limit: float = DEFAULT_COST_LIMIT_USD, seeds=SEEDS, rounds: int = ROUNDS,
        per_round: int = PER_ROUND, workers: int = 3, n_boot: int = N_BOOT,
        exp1_path: Path = RESULTS / "experiment1.json", log=print) -> dict:
    started = time.perf_counter()
    cache, model = cache or ResponseCache(), model or os.environ["CLAUDE_MODEL"]
    d = inputs[0]
    estimate = estimate_worst_case(d, client, model, seeds, rounds)
    print(f"[estimate] worst case ${estimate['worst_case_usd']:.2f} ({estimate['worst_case_calls']} calls, nothing "
          f"cached, max clusters and attempts every round); hard stop at ${cost_limit:.2f} before every step")
    budget = Budget(cost_limit)
    ds = run_dataset(d, client, cache, model, budget, seeds, rounds, per_round, workers, n_boot, out_dir, log)
    exp1 = json.loads(exp1_path.read_text(encoding="utf-8"))["datasets"][d.name]["crowds"]
    report = {"settings": {"model": model, "seeds": list(seeds), "rounds": rounds, "per_round": per_round,
                           "n_boot": n_boot, "cost_limit_usd": cost_limit, "git_commit": _git_commit()},
              "estimate": estimate, "cost": {"actual_usd": round(budget.spent, 4)}, "datasets": {d.name: ds},
              "references": {c: exp1[c] for c in REFERENCE_ORDER if c in exp1},
              "runtime_seconds": round(time.perf_counter() - started, 1)}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "experiment2b.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (out_dir / "experiment2b.md").write_text(to_markdown(report), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 2b: redesigned loop on GSS")
    parser.add_argument("--max-cost", type=float, default=DEFAULT_COST_LIMIT_USD)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    import anthropic
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    inputs = [d for d in load_inputs() if d.name == "gss"]
    try:
        run(inputs, anthropic.Anthropic(), cost_limit=args.max_cost, seeds=tuple(args.seeds), workers=args.workers)
    except CostLimitExceeded as e:
        raise SystemExit(f"STOPPED by cost guard: {e}") from None
    print((RESULTS / "experiment2b.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
