"""Experiment 1 (H3): plain Claude persona crowds vs the H2 crowds.

`python -m blindspot.eval.experiment1 [--estimate-only] [--max-cost 5]`
Target = TEST split for both datasets; holdout dropped at load (via experiment0.load_inputs).
The cost guard runs over both datasets before any paid call.
Writes results/experiment1.json, results/experiment1.md and GSS maps for the Claude crowds.
"""
import argparse
import dataclasses
import json
import os
import time
from pathlib import Path

import numpy as np

from blindspot.coverage import evaluate, family_entropy, real_distances, top_share
from blindspot.data.schema import Item, Respondents
from blindspot.generators import claude_personas as cp
from blindspot.generators.claude_client import (DEFAULT_COST_LIMIT_USD, CostLimitExceeded, ResponseCache,
                                                Usage, check_cost)
from blindspot.space import map_coords

from .experiment0 import (K, K_VALUES, N_BOOT, SEED, DatasetInput, _assert_no_holdout, _fmt_ci, _git_commit,
                          build_crowds, load_inputs)

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
H2_JSON = RESULTS / "experiment0.json"
BUNCHING_THRESHOLD = 0.80


def bunching_share(resp: Respondents, threshold: float = BUNCHING_THRESHOLD) -> float:
    """Share of items where more than `threshold` of people give the same option."""
    tops = [top_share(resp.answers[it.code]) for it in resp.items]
    return float(np.mean([t > threshold for t in tops]))


def unweighted(items: list[Item]) -> list[Item]:
    """Every item its own family -> equal item weights in Gower."""
    return [dataclasses.replace(it, family=it.code) for it in items]


def _target(d: DatasetInput) -> Respondents:
    return d.real.subset(d.splits["test"])


def make_plans(inputs: list[DatasetInput], model: str, recipe: str = "") -> dict[str, cp.Plan]:
    return {d.name: cp.plan(d.name, _target(d).demographics, _target(d).items, model, recipe) for d in inputs}


def guard(plans: dict[str, cp.Plan], client, cache: ResponseCache, limit: float, batch: bool) -> dict:
    per = {name: cp.estimate(p, client, cache, batch) for name, p in plans.items()}
    for name, e in per.items():
        print(f"[estimate] {name}: {e['calls']} uncached calls, ~${e['cost_usd']:.2f}")
    total = {"calls": sum(e["calls"] for e in per.values()),
             "input_tokens": sum(e["input_tokens"] for e in per.values()),
             "output_tokens": sum(e["output_tokens"] for e in per.values()),
             "model": next(iter(plans.values())).model, "batch": batch,
             "cost_usd": round(sum(e["cost_usd"] for e in per.values()), 4)}
    check_cost(total, limit)
    return {"per_dataset": per, "total": total, "limit_usd": limit}


def cost_of_all_responses(plans: dict[str, cp.Plan], cache: ResponseCache, batch: bool) -> dict:
    """Billed cost of every cached response this run uses (first attempts + retries), including calls
    paid in earlier, interrupted runs. Priced as batch calls when `batch` is set."""
    from blindspot.generators.claude_client import token_cost

    total = Usage(batch=batch)
    for p in plans.values():
        keys = [s.key for s in p.specs] + [p.retry_spec(i).key for i in range(len(p.specs))]
        for key in keys:
            rec = cache.get(key)
            if rec is not None:
                total.add(rec["usage"], p.model, batch)
    out = total.to_dict()
    out.pop("errors", None)
    return out


def _public(m: dict) -> dict:
    """Drop private keys and respondent ids; keep only how many real people are uncovered."""
    out = {k: v for k, v in m.items() if not k.startswith("_") and k != "uncovered_ids"}
    out["n_uncovered"] = len(m["uncovered_ids"])
    return out


def evaluate_dataset(d: DatasetInput, crowds: cp.ClaudeCrowds, h2: dict, n_boot: int, seed: int,
                     maps: bool, out_dir: Path) -> dict:
    target = _target(d)
    if h2["n_target"] != len(target.answers):
        raise ValueError(f"{d.name}: H2 target size {h2['n_target']} != {len(target.answers)}; split changed?")
    rr = real_distances(target, target.items)
    new = {"claude_mode": crowds.mode, "claude_sampled": crowds.sampled}
    results = {n: evaluate(target, c, target.items, k=K, n_boot=n_boot, seed=seed, k_values=K_VALUES,
                           real_dist=rr) for n, c in new.items()}
    for name, m in results.items():                       # check the full output before ids are dropped
        _assert_no_holdout(m["uncovered_ids"], d.splits, f"{d.name} output {name}")
    map_files = {}
    if maps and d.name == "gss":
        flags = {n: {"real_covered": results[n]["_real_covered"],
                     "synthetic_inside": results[n]["_synthetic_inside"]} for n in new}
        paths = map_coords(target, new, seed=seed, covered_by_name=flags, dataset=d.name, out_dir=out_dir / "maps")
        map_files = {n: f"maps/{p.name}" for n, p in paths.items()}
    bunching = {"real_target": bunching_share(target), **{n: bunching_share(c) for n, c in new.items()}}
    if "gpt41mini_twin" in d.fixed_crowds:
        bunching["gpt41mini_twin"] = bunching_share(d.fixed_crowds["gpt41mini_twin"])
    missing = int(crowds.mode.answers.isna().sum().sum())
    return {
        "title": h2["title"], "n_target": h2["n_target"], "n_pool": h2["n_pool"],
        "real_family_entropy": family_entropy(target),
        "crowds": {**h2["crowds"], **{n: _public(m) for n, m in results.items()}},
        "new_crowds": list(new),
        "bunching_gt_80": bunching,
        "failures": {"people_with_missing_items": len(crowds.failures), "missing_answers": missing,
                     "missing_share": missing / (len(target.answers) * len(target.items)),
                     "examples": dict(list(crowds.failures.items())[:5])},
        "map_files": map_files,
    }


def robustness_twin(d: DatasetInput, crowds: cp.ClaudeCrowds, seed: int, n_boot: int) -> dict:
    """Main Twin table recomputed with unweighted Gower: does the coverage ranking change?"""
    target = _target(d)
    pool = d.real.subset(d.splits["pool"])
    all_crowds = {**build_crowds(d, target, pool, seed), "claude_mode": crowds.mode,
                  "claude_sampled": crowds.sampled}
    out = {}
    for label, items in (("family_weighted", target.items), ("unweighted", unweighted(target.items))):
        rr = real_distances(target, items)
        out[label] = {n: _public(evaluate(target, c, items, k=K, n_boot=n_boot, seed=seed, k_values=(K,),
                                          real_dist=rr)) for n, c in all_crowds.items()}
    rank = {label: sorted(res, key=lambda n: -res[n]["coverage"]) for label, res in out.items()}
    return {"coverage": {label: {n: m["coverage"] for n, m in res.items()} for label, res in out.items()},
            "coverage_ci": {label: {n: m["coverage_ci"] for n, m in res.items()} for label, res in out.items()},
            "ranking": rank, "ranking_changed": rank["family_weighted"] != rank["unweighted"]}


def run(inputs: list[DatasetInput], client, out_dir: Path = RESULTS, cache: ResponseCache | None = None,
        model: str | None = None, cost_limit: float = DEFAULT_COST_LIMIT_USD, batch: bool = True,
        n_boot: int = N_BOOT, seed: int = SEED, maps: bool = True, h2_path: Path = H2_JSON,
        estimate_only: bool = False, prompt_dir: Path = cp.PROMPT_DIR) -> dict:
    started = time.perf_counter()
    cache = cache or ResponseCache()
    model = model or os.environ["CLAUDE_MODEL"]
    h2 = json.loads(Path(h2_path).read_text(encoding="utf-8"))
    for d in inputs:
        _assert_no_holdout(d.real.answers.index, d.splits, f"{d.name} real data")
    plans = make_plans(inputs, model)
    estimate = guard(plans, client, cache, cost_limit, batch)
    if estimate_only:
        return {"estimate": estimate}
    for p in plans.values():
        cp.save_prompts(p, prompt_dir)
    usage = Usage(batch=batch)
    generated = {d.name: cp.execute(plans[d.name], client, cache, usage, seed=seed, batch=batch,
                                    cost_limit=cost_limit) for d in inputs}
    datasets = {d.name: evaluate_dataset(d, generated[d.name], h2["datasets"][d.name], n_boot, seed, maps, out_dir)
                for d in inputs}
    twin = next((d for d in inputs if d.name == "twin2k"), None)
    report = {
        "settings": {**h2["settings"], "git_commit": _git_commit(), "model": model,
                     "prompt_version": cp.PROMPT_VERSION, "recipe": "", "batch": batch,
                     "thinking": "off (model-specific: see claude_personas.thinking_off)", "sampling": "API default temperature (1.0)",
                     "h2_rows_from": Path(h2_path).name},
        "cost": {"estimate": estimate, "actual": usage.to_dict(),
                 "all_responses_used": cost_of_all_responses(plans, cache, batch)},
        "datasets": datasets,
        "robustness_twin": robustness_twin(twin, generated[twin.name], seed, n_boot) if twin else None,
        "runtime_seconds": round(time.perf_counter() - started, 1),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "experiment1.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (out_dir / "experiment1.md").write_text(to_markdown(report), encoding="utf-8")
    return report


# ---------- markdown ----------
def _sorted_crowds(ds: dict) -> list[tuple[str, dict]]:
    return sorted(ds["crowds"].items(), key=lambda kv: -kv[1]["coverage"])


def _main_table(ds: dict) -> list[str]:
    lines = ["| crowd | n | coverage [95% CI] | precision [95% CI] | rcg_ratio |", "| --- | --- | --- | --- | --- |"]
    for n, m in _sorted_crowds(ds):
        name = f"**{n}**" if n in ds["new_crowds"] else n
        lines.append(f"| {name} | {m['n_synthetic']} | {_fmt_ci(m['coverage'], m['coverage_ci'])} | "
                     f"{_fmt_ci(m['precision'], m['precision_ci'])} | {m['rcg_ratio']:.3f} |")
    return lines


def _k_table(ds: dict, ks: list[int]) -> list[str]:
    lines = ["| crowd | " + " | ".join(f"k={k}" for k in ks) + " |", "| --- |" + " --- |" * len(ks)]
    lines += [f"| {n} | " + " | ".join(f"{m['k_sensitivity'][str(k)]['coverage']:.3f} / "
                                      f"{m['k_sensitivity'][str(k)]['precision']:.3f}" for k in ks) + " |"
              for n, m in _sorted_crowds(ds)]
    return lines


def _entropy_table(ds: dict) -> list[str]:
    names = [n for n, _ in _sorted_crowds(ds)]
    lines = ["| family | real target | " + " | ".join(names) + " |", "| --- |" + " --- |" * (len(names) + 1)]
    for fam, real_h in ds["real_family_entropy"].items():
        lines.append(f"| {fam} | {real_h:.2f} | " + " | ".join(
            f"{ds['crowds'][n]['per_family_entropy']['crowd'][fam]:.2f}" for n in names) + " |")
    return lines


def _robustness_md(rob: dict) -> list[str]:
    w, u = rob["coverage"]["family_weighted"], rob["coverage"]["unweighted"]
    lines = ["## Robustness: Twin with unweighted Gower (every item equal)", "",
             "| crowd | coverage, family-weighted | coverage, unweighted |", "| --- | --- | --- |"]
    lines += [f"| {n} | {w[n]:.3f} | {u[n]:.3f} |" for n in rob["ranking"]["family_weighted"]]
    return lines + ["", f"Ranking, family-weighted: {' > '.join(rob['ranking']['family_weighted'])}",
                    f"Ranking, unweighted: {' > '.join(rob['ranking']['unweighted'])}",
                    f"Ranking changed: **{'yes' if rob['ranking_changed'] else 'no'}**", ""]


def to_markdown(r: dict) -> str:
    s, c = r["settings"], r["cost"]
    act, est = c["actual"], c["estimate"]["total"]
    lines = ["# Experiment 1: plain Claude persona crowds", "",
             f"Model `{s['model']}`, prompt {s['prompt_version']}, empty recipe, thinking off, API-default "
             f"sampling, {'Message Batches' if s['batch'] else 'standard'} API. Target = test split; holdout "
             f"untouched. H2 rows copied from `{s['h2_rows_from']}`; new rows in bold. k = {s['k']}, "
             f"{s['n_boot']} bootstrap resamples. Commit `{s['git_commit']}`. Runtime {r['runtime_seconds']} s.", "",
             f"Cost: estimate ${est['cost_usd']:.2f} for {est['calls']} uncached calls (limit "
             f"${c['estimate']['limit_usd']:.2f}); actual ${act['cost_usd']:.2f} for {act['calls']} calls, "
             f"{act['input_tokens']:,} input + {act['cache_write_tokens']:,} cache-write + "
             f"{act['cache_read_tokens']:,} cache-read + {act['output_tokens']:,} output tokens; "
             f"{len(act['errors'])} API errors.",
             f"Total cost of every response this experiment uses (all runs, incl. interrupted ones and "
             f"retries): ${c['all_responses_used']['cost_usd']:.2f} for {c['all_responses_used']['calls']} "
             f"responses.", ""]
    for ds in r["datasets"].values():
        f = ds["failures"]
        lines += [f"## {ds['title']} (target n = {ds['n_target']}, pool n = {ds['n_pool']})", ""]
        lines += _main_table(ds)
        lines += ["", "k-sensitivity (coverage / precision):", ""] + _k_table(ds, s["k_values"])
        lines += ["", "Per-family mean normalized entropy:", ""] + _entropy_table(ds)
        lines += ["", f"Share of items with more than {BUNCHING_THRESHOLD:.0%} of people on one option: "
                  + ", ".join(f"{n} {v:.2f}" for n, v in ds["bunching_gt_80"].items()),
                  f"Parsing: {f['people_with_missing_items']} synthetic people with a missing item after retry; "
                  f"{f['missing_answers']} missing answers ({f['missing_share']:.2%}).", ""]
    if r["robustness_twin"]:
        lines += _robustness_md(r["robustness_twin"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 1: Claude persona crowds")
    parser.add_argument("--max-cost", type=float, default=DEFAULT_COST_LIMIT_USD, help="cost guard limit in USD")
    parser.add_argument("--estimate-only", action="store_true", help="print the cost estimate and exit")
    parser.add_argument("--no-batch", action="store_true", help="use bounded concurrency instead of batches")
    args = parser.parse_args()

    import anthropic
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    client = anthropic.Anthropic()
    try:
        report = run(load_inputs(), client, cost_limit=args.max_cost, batch=not args.no_batch,
                     estimate_only=args.estimate_only)
    except CostLimitExceeded as e:
        raise SystemExit(f"STOPPED by cost guard: {e}") from None
    if args.estimate_only:
        print(json.dumps(report["estimate"], indent=1))
        return
    print((RESULTS / "experiment1.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
