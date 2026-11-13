"""H4d-big (zero cost): H4d repeated with candidates from the cached Sonnet 5.5 and Opus 5.5 probabilities.

`python -m blindspot.eval.h4d_big`
GSS, 20 live items, TEST split for scoring, holdout never used, NO API calls (the model responses come from the
cache written by experiment 1b; the client is None, so a cache miss would fail instead of spending money).
Identical to H4d: 10 seeds, the same asked pool people per seed (same seeded permutation), budgets 25/50/75/100,
realism filter = only the asked people (k = min(5, B//5)), lambda chosen per seed by the half-split from
{0, .25, .5, 1}, then rebuilt with all asked people. Compared with copula_on_asked on the same asked people and the
Haiku select_claude rows saved by H4d.
"""
import json
import time
from pathlib import Path

import numpy as np

from blindspot.coverage import real_distances
from blindspot.generators import claude_personas as cp
from blindspot.generators.baselines import gaussian_copula
from blindspot.generators.claude_client import ResponseCache

from .bakeoff1 import cached_parsed, draw_candidates, make_crowd
from .experiment0 import SEED, _assert_no_holdout, _git_commit, load_inputs
from .experiment1b import OPUS, SONNET
from .experiment_fp import _agg, restrict
from .h4d import BUDGETS, LAMBDAS, N_SEEDS, build_crowd, choose_lambda, score

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
MODELS = {"opus": OPUS, "sonnet": SONNET}


def run_model(parsed, ctx: dict) -> list[dict]:
    """Per seed and budget: select_claude (big-model candidates) and copula_on_asked on the same asked people."""
    target, pool, items, splits = ctx["target"], ctx["pool"], ctx["items"], ctx["splits"]
    per_seed = []
    for i in range(N_SEEDS):
        seed = SEED + i
        cands = np.array(draw_candidates(parsed, items, seed))
        order = np.array(pool.answers.index)[np.random.default_rng(seed).permutation(len(pool.answers))]
        rows = {}
        for b in BUDGETS:
            asked = pool.subset(list(order[:b]))
            lam = choose_lambda(cands, asked, items, None, seed)
            raw, _ = build_crowd(cands, asked, lam, items, None)
            crowd = make_crowd(raw, target.demographics, items, "select")
            cop = gaussian_copula(asked, target.demographics, splits, seed)
            rows[b] = {"select": score(target, crowd, ctx["rr"], seed, crowd, asked), "lambda": lam,
                       "copula": score(target, cop, ctx["rr"], seed, cop, asked)}
        per_seed.append(rows)
    return per_seed


def aggregate(per_seed: list[dict]) -> dict:
    out = {}
    for b in BUDGETS:
        rows = [s[b] for s in per_seed]
        d = np.array([r["select"]["coverage"] - r["copula"]["coverage"] for r in rows])
        entry = {"chosen_lambdas": {str(lam): [r["lambda"] for r in rows].count(lam) for lam in LAMBDAS},
                 "paired_vs_copula": {"mean_diff": float(d.mean()), "sd": float(d.std(ddof=1)),
                                      "wins": int((d > 0).sum()), "n_seeds": len(d)}}
        for m in ("select", "copula"):
            sc = [r[m] for r in rows]
            entry[m] = {"coverage": _agg([x["coverage"] for x in sc]), "precision": _agg([x["precision"] for x in sc]),
                        "near_dup_vs_asked": _agg([x["near_dup_vs_asked"] for x in sc]),
                        "k_sensitivity": {kk: {s: float(np.mean([x["k_sensitivity"][kk][s] for x in sc]))
                                               for s in ("coverage", "precision")} for kk in ("3", "5", "10")}}
        out[str(b)] = entry
    return out


def haiku_reference() -> dict:
    h = json.loads((RESULTS / "h4d.json").read_text(encoding="utf-8"))
    paired = {(p["budget"], p["a"], p["b"]): p for p in h["paired"]}
    ref = {}
    for b in BUDGETS:
        sel = h["results"][str(b)]["select_claude"]
        p = paired[(b, "select_claude", "copula_on_asked")]
        ref[str(b)] = {"select": sel, "copula": h["results"][str(b)]["copula_on_asked"],
                       "paired_vs_copula": {"mean_diff": p["mean_diff"], "sd": p["sd"], "wins": p["wins"],
                                            "n_seeds": p["n_seeds"]}, "chosen_lambdas": sel["chosen_lambdas"]}
    return ref


def run(out_dir: Path = RESULTS) -> dict:
    from dotenv import dotenv_values

    started = time.perf_counter()
    gss_in = next(d for d in load_inputs() if d.name == "gss")               # holdout already dropped
    _assert_no_holdout(gss_in.real.answers.index, gss_in.splits, "gss real data")
    live = [i.code for i in gss_in.real.items if i.live_ok]
    full_target, full_pool = (gss_in.real.subset(gss_in.splits[s]) for s in ("test", "pool"))
    target, pool = restrict(full_target, live), restrict(full_pool, live)
    ctx = {"target": target, "pool": pool, "items": target.items, "splits": gss_in.splits,
           "rr": real_distances(target, target.items)}
    cache = ResponseCache()
    results = {}
    for key, model in MODELS.items():
        plan = cp.plan("gss", full_target.demographics, full_target.items, model)
        misses = sum(cache.get(s.key) is None for s in plan.specs)
        if misses:
            raise SystemExit(f"{misses} {model} responses are not cached; run experiment1b first (no API calls here)")
        results[key] = aggregate(run_model(cached_parsed(plan, cache), ctx))
    haiku = haiku_reference()
    for b in BUDGETS:   # same asked people and copula: our copula numbers must equal H4d's
        mine = results["opus"][str(b)]["copula"]["coverage"]["mean"]
        assert abs(mine - haiku[str(b)]["copula"]["coverage"]["mean"]) < 1e-9, f"copula mismatch at B={b}"
    report = {"settings": {"budgets": list(BUDGETS), "n_seeds": N_SEEDS, "lambdas": list(LAMBDAS), "k": 5,
                           "n_items": len(live), "n_target": len(target.answers), "n_pool": len(pool.answers),
                           "models": MODELS, "api_calls": 0, "git_commit": _git_commit(),
                           "near_dup_basis": "synthetic rows within 0.02 Gower of any ASKED person"},
              "haiku_from_h4d": haiku, "big_models": results,
              "runtime_seconds": round(time.perf_counter() - started, 1)}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "h4d_big.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (out_dir / "h4d_big.md").write_text(to_markdown(report), encoding="utf-8")
    return report


def _c(a: dict) -> str:
    return f"{a['mean']:.3f} ({a['min']:.3f}-{a['max']:.3f})"


def to_markdown(r: dict) -> str:
    s, big, hk = r["settings"], r["big_models"], r["haiku_from_h4d"]
    rows = [("select_claude, Opus 5.5 probs", lambda b: big["opus"][str(b)]["select"]),
            ("select_claude, Sonnet 5.5 probs", lambda b: big["sonnet"][str(b)]["select"]),
            ("select_claude, Haiku 5.5 probs (H4d)", lambda b: hk[str(b)]["select"]),
            ("copula_on_asked (same asked people)", lambda b: big["opus"][str(b)]["copula"])]
    lines = ["# H4d-big: select from Sonnet / Opus candidates with B asked people as the realism filter", "",
             f"Zero API calls (cached responses from experiment 1b), GSS {s['n_items']} live items, test n = {s['n_target']}, "
             f"{s['n_seeds']} seeds, same asked people and settings as H4d (k = min(5, B//5), lambda by half-split from "
             f"{s['lambdas']}). Cells: mean (min-max over seeds). Haiku rows are H4d's saved numbers. Near-dup = synthetic "
             f"rows within 0.02 Gower of an asked person. Commit `{s['git_commit']}`. Runtime {r['runtime_seconds']} s.", ""]
    for title, key in (("Coverage", "coverage"), ("Precision", "precision"), ("Near-duplicate share", "near_dup_vs_asked")):
        lines += [f"## {title}", "", "| method | " + " | ".join(f"B={b}" for b in s["budgets"]) + " |",
                  "| --- |" + " --- |" * len(s["budgets"])]
        lines += [f"| {name} | " + " | ".join(_c(get(b)[key]) for b in s["budgets"]) + " |" for name, get in rows]
        lines += [""]
    lines += ["## Paired vs copula_on_asked (coverage, same seeds and asked people)", "",
              "| budget | model | mean diff (select - copula) | sd | select wins |", "| --- | --- | --- | --- | --- |"]
    for b in s["budgets"]:
        for name, p in (("Opus", big["opus"][str(b)]["paired_vs_copula"]), ("Sonnet", big["sonnet"][str(b)]["paired_vs_copula"]),
                        ("Haiku (H4d)", hk[str(b)]["paired_vs_copula"])):
            lines.append(f"| {b} | {name} | {p['mean_diff']:+.3f} | {p['sd']:.3f} | {p['wins']}/{p['n_seeds']} |")
    lines += ["", "## Chosen lambdas (count of seeds)", "", "| model | B | " + " | ".join(f"λ={x}" for x in s["lambdas"]) + " |",
              "| --- | --- |" + " --- |" * len(s["lambdas"])]
    for name, src in (("Opus", lambda b: big["opus"][str(b)]), ("Sonnet", lambda b: big["sonnet"][str(b)]), ("Haiku", lambda b: hk[str(b)])):
        for b in s["budgets"]:
            lines.append(f"| {name} | {b} | " + " | ".join(str(src(b)["chosen_lambdas"][str(x)]) for x in s["lambdas"]) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    run()
    print((RESULTS / "h4d_big.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
