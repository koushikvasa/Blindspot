"""H4d (zero cost): with only B real people, does selecting from cached AI candidates beat statistics fit on them?

`python -m blindspot.eval.h4d`
GSS, 20 live items, TEST split for scoring, holdout never touched, no Claude or Flashpoint calls.
B = 25/50/75/100 random pool people per seed (same asked people for every method), 10 seeds.

select_claude          50 candidates per test persona from cached claude_sampled probabilities; realism = mean
                       Gower to the k nearest ASKED people (k = min(5, B//5)); greedy pick of the candidate
                       minimising realism - lambda*novelty. Lambda per seed: split the asked people in half,
                       build with the first half as the filter for lambda in {0, .25, .5, 1}, keep the lambda
                       that covers most of the held-out half (radius = k-th nearest other ASKED person), then
                       rebuild with all asked people.
select_claude_plus_fp  same, but the 400 saved Flashpoint people join a SHARED pool: in every slot they compete
                       with that persona's 50 Claude candidates (same realism and novelty), each Flashpoint person
                       can fill at most one slot in the crowd.
copula_on_asked        Gaussian copula fit on the same B people.
claude_plus_asked      claude_mode with the B asked people swapped in (INCLUDES REAL RESPONDENTS).
select_plus_asked      select_claude crowd with the B asked people swapped in (INCLUDES REAL RESPONDENTS).
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from blindspot.agent.guards import near_duplicates
from blindspot.coverage import evaluate, real_distances
from blindspot.data.schema import Respondents
from blindspot.generators import claude_personas as cp
from blindspot.generators.baselines import gaussian_copula
from blindspot.generators.claude_client import ResponseCache, Usage
from blindspot.generators.flashpoint_panel import load as load_flashpoint
from blindspot.space import encode, gower

from .bakeoff1 import N_CANDIDATES, _novelty, _scale, cached_parsed, draw_candidates, make_crowd
from .experiment0 import K, K_VALUES, SEED, _assert_no_holdout, _git_commit, load_inputs
from .experiment2 import swap_in_asked
from .experiment_fp import FP_DIR, RUNS, _agg, claude_mode_full, restrict

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
BUDGETS = (25, 50, 75, 100)
N_SEEDS = 10
LAMBDAS = (0.0, 0.25, 0.5, 1.0)
N_BOOT = 20
METHODS = ("select_claude", "select_claude_plus_fp", "copula_on_asked", "claude_plus_asked", "select_plus_asked")
PAIRS = (("select_claude", "copula_on_asked"), ("select_claude_plus_fp", "copula_on_asked"),
         ("select_plus_asked", "claude_plus_asked"))


def realism_k(raw: np.ndarray, ref: Respondents, k: int) -> np.ndarray:
    """Mean Gower distance to the k nearest people of `ref` (lower = more realistic)."""
    d = np.nan_to_num(gower(raw / _scale(ref.items), encode(ref), ref.items), nan=1.0)
    return np.partition(d, k - 1, axis=1)[:, :k].mean(axis=1)


def k_for(n_ref: int) -> int:
    return max(1, min(5, n_ref // 5))


def build_crowd(cands: np.ndarray, ref: Respondents, lam: float, items, fp_raw: np.ndarray | None = None
                ) -> tuple[np.ndarray, int]:
    """cands: personas x 50 x items. Greedy per persona; returns (raw answers, number of Flashpoint people used)."""
    n, per, m = cands.shape
    k = k_for(len(ref.answers))
    real = realism_k(cands.reshape(-1, m), ref, k).reshape(n, per)
    scale, chosen, fp_used = _scale(items), [], 0
    if fp_raw is not None:
        fp_real, fp_enc = realism_k(fp_raw, ref, k), fp_raw / scale
        taken, nov_fp = np.zeros(len(fp_raw), bool), np.zeros(len(fp_raw))
    for c, r in zip(cands, real):
        nov = _novelty(c / scale, np.array(chosen) / scale if chosen else np.empty((0, m)), items)
        score = r - lam * nov
        j = int(np.argmin(score))
        pick = c[j]
        if fp_raw is not None:
            fs = np.where(taken, np.inf, fp_real - lam * nov_fp)
            jf = int(np.argmin(fs))
            if fs[jf] < score[j]:
                pick, taken[jf], fp_used = fp_raw[jf], True, fp_used + 1
        chosen.append(pick)
        if fp_raw is not None:
            d = np.nan_to_num(gower(fp_enc, (pick / scale)[None], items), nan=1.0)[:, 0]
            nov_fp = d if len(chosen) == 1 else np.minimum(nov_fp, d)
    return np.array(chosen), fp_used


def asked_radius(asked: Respondents) -> np.ndarray:
    """Distance from each asked person to their k-th nearest OTHER asked person (their own k-NN ball)."""
    enc = encode(asked)
    d = np.nan_to_num(gower(enc, enc, asked.items), nan=np.inf)
    np.fill_diagonal(d, np.inf)
    return np.partition(d, k_for(len(asked.answers)) - 1, axis=1)[:, k_for(len(asked.answers)) - 1]


def choose_lambda(cands, asked: Respondents, items, fp_raw, seed: int) -> float:
    """Lambda covering most of the held-out half of the asked people (radii from all asked people)."""
    rng = np.random.default_rng([seed, len(asked.answers), 7])
    ids = np.array(asked.answers.index)[rng.permutation(len(asked.answers))]
    half = len(ids) // 2
    filt, held = asked.subset(list(ids[:half])), asked.subset(list(ids[half:]))
    radius = pd.Series(asked_radius(asked), index=asked.answers.index).loc[held.answers.index].to_numpy()
    held_enc, best, best_cov = encode(held), LAMBDAS[0], -1.0
    for lam in LAMBDAS:
        raw, _ = build_crowd(cands, filt, lam, items, fp_raw)
        nn = np.nan_to_num(gower(held_enc, raw / _scale(items), items), nan=np.inf).min(axis=1)
        cov = float((nn <= radius).mean())
        if cov > best_cov + 1e-12:
            best, best_cov = lam, cov
    return best


def score(target, crowd, rr, seed: int, synthetic: Respondents, asked: Respondents) -> dict:
    m = evaluate(target, crowd, target.items, k=K, n_boot=N_BOOT, seed=seed, k_values=K_VALUES, real_dist=rr)
    return {"coverage": m["coverage"], "precision": m["precision"], "k_sensitivity": m["k_sensitivity"],
            "near_dup_vs_asked": near_duplicates(synthetic, asked).share}


def run_seed(seed: int, ctx: dict) -> dict:
    target, pool, items, splits = ctx["target"], ctx["pool"], ctx["items"], ctx["splits"]
    cands = np.array(draw_candidates(ctx["parsed"], items, seed))            # personas x 50 x items
    order = np.array(pool.answers.index)[np.random.default_rng(seed).permutation(len(pool.answers))]
    out = {}
    for b in BUDGETS:
        asked = pool.subset(list(order[:b]))
        res, extra = {}, {}
        sel = {}
        for name, fp in (("select_claude", None), ("select_claude_plus_fp", ctx["fp_raw"])):
            lam = choose_lambda(cands, asked, items, fp, seed)
            raw, used = build_crowd(cands, asked, lam, items, fp)
            crowd = make_crowd(raw, target.demographics, items, name)
            sel[name] = crowd
            res[name] = score(target, crowd, ctx["rr"], seed, crowd, asked)
            extra[name] = {"lambda": lam, "fp_share": used / len(raw)}
        cop = gaussian_copula(asked, target.demographics, splits, seed)
        res["copula_on_asked"] = score(target, cop, ctx["rr"], seed, cop, asked)
        for name, base in (("claude_plus_asked", ctx["base"]), ("select_plus_asked", sel["select_claude"])):
            crowd = swap_in_asked(base, asked)
            synthetic = base.subset(list(base.answers.index[b:]))             # swapped-in rows are exempt
            res[name] = score(target, crowd, ctx["rr"], seed, synthetic, asked)
        out[b] = {"scores": res, "extra": extra}
    return out


def aggregate(per_seed: list[dict]) -> dict:
    out = {}
    for b in BUDGETS:
        rows = [s[b] for s in per_seed]
        out[str(b)] = {}
        for m in METHODS:
            sc = [r["scores"][m] for r in rows]
            entry = {"coverage": _agg([x["coverage"] for x in sc]), "precision": _agg([x["precision"] for x in sc]),
                     "near_dup_vs_asked": _agg([x["near_dup_vs_asked"] for x in sc]),
                     "k_sensitivity": {kk: {s: float(np.mean([x["k_sensitivity"][kk][s] for x in sc]))
                                            for s in ("coverage", "precision")} for kk in ("3", "5", "10")}}
            if m in rows[0]["extra"]:
                lams = [r["extra"][m]["lambda"] for r in rows]
                entry["chosen_lambdas"] = {str(lam): lams.count(lam) for lam in LAMBDAS}
                entry["fp_share_of_crowd"] = float(np.mean([r["extra"][m]["fp_share"] for r in rows]))
            out[str(b)][m] = entry
    return out


def paired(per_seed: list[dict]) -> list[dict]:
    rows = []
    for b in BUDGETS:
        for a, c in PAIRS:
            d = np.array([s[b]["scores"][a]["coverage"] - s[b]["scores"][c]["coverage"] for s in per_seed])
            rows.append({"budget": b, "a": a, "b": c, "mean_diff": float(d.mean()), "sd": float(d.std(ddof=1)),
                         "wins": int((d > 0).sum()), "n_seeds": len(d)})
    return rows


def references() -> dict:
    bk = json.loads((RESULTS / "bakeoff1.json").read_text(encoding="utf-8"))["A_gss_select_from_many_n334"]
    names = ["claude_mode", "select_from_50_lambda_0.0", "select_from_50_lambda_0.5", "select_from_50_lambda_1.0",
             "gaussian_copula", "other_real_people"]
    return {n: {"coverage": bk[n]["coverage"]["mean"], "precision": bk[n]["precision"]["mean"]} for n in names}


def run(out_dir: Path = RESULTS) -> dict:
    started = time.perf_counter()
    gss_in = next(d for d in load_inputs() if d.name == "gss")               # holdout already dropped
    _assert_no_holdout(gss_in.real.answers.index, gss_in.splits, "gss real data")
    live = [i.code for i in gss_in.real.items if i.live_ok]
    full_target, full_pool = (gss_in.real.subset(gss_in.splits[s]) for s in ("test", "pool"))
    target, pool = restrict(full_target, live), restrict(full_pool, live)
    from dotenv import dotenv_values

    plan = cp.plan("gss", full_target.demographics, full_target.items, dotenv_values(ROOT / ".env")["CLAUDE_MODEL"])
    cache = ResponseCache()
    base = claude_mode_full(full_target, live)
    parsed = cached_parsed(plan, cache)
    meta = json.loads((FP_DIR / "runs_meta.json").read_text(encoding="utf-8"))
    fp_raw = np.vstack([load_flashpoint(FP_DIR / meta["runs"][t]["rows_file"], target.items, t)[0].answers.to_numpy()
                        for t in RUNS])
    ctx = {"target": target, "pool": pool, "items": target.items, "splits": gss_in.splits, "parsed": parsed,
           "fp_raw": fp_raw, "base": base, "rr": real_distances(target, target.items)}
    per_seed = [run_seed(SEED + i, ctx) for i in range(N_SEEDS)]
    report = {
        "settings": {"budgets": list(BUDGETS), "n_seeds": N_SEEDS, "lambdas": list(LAMBDAS),
                     "n_candidates": N_CANDIDATES, "k_realism": "min(5, B//5)", "k": K, "n_boot": N_BOOT,
                     "n_items": len(live), "n_target": len(target.answers), "n_pool": len(pool.answers),
                     "n_flashpoint_candidates": len(fp_raw), "near_dup_threshold": 0.02,
                     "near_dup_basis": "synthetic rows within 0.02 Gower of any ASKED person (swapped-in rows exempt)",
                     "git_commit": _git_commit()},
        "claude_mode_budget_0": references()["claude_mode"], "references_n334_20items": references(),
        "results": aggregate(per_seed), "paired": paired(per_seed),
        "runtime_seconds": round(time.perf_counter() - started, 1)}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "h4d.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (out_dir / "h4d.md").write_text(to_markdown(report), encoding="utf-8")
    return report


def _cell(a: dict) -> str:
    return f"{a['mean']:.3f} ({a['p2_5']:.3f}-{a['p97_5']:.3f})"


def to_markdown(r: dict) -> str:
    s, res = r["settings"], r["results"]
    lines = ["# H4d: B real people, select from AI candidates vs fit statistics (zero cost, GSS test split)", "",
             f"{s['n_items']} live items, target n = {s['n_target']}, {s['n_seeds']} seeds, same asked people for every "
             f"method. Cells: mean (2.5-97.5 percentile across seeds). k = {s['k']}. Realism filter = the B asked "
             f"people only (k = {s['k_realism']}); lambda chosen per seed from {s['lambdas']} on a held-out half of the "
             f"asked people. `select_claude_plus_fp`: the {s['n_flashpoint_candidates']} saved Flashpoint people form a "
             "shared pool competing in every slot with that persona's 50 Claude candidates, each used at most once. "
             "`claude_plus_asked` and `select_plus_asked` INCLUDE REAL RESPONDENTS (asked people swapped in). Near-dup = "
             f"{s['near_dup_basis']}. Commit `{s['git_commit']}`. Runtime {r['runtime_seconds']} s.", ""]
    for title, key in (("Coverage", "coverage"), ("Precision", "precision"), ("Near-duplicate share", "near_dup_vs_asked")):
        lines += [f"## {title}", "", "| method | " + " | ".join(f"B={b}" for b in s["budgets"]) + " |",
                  "| --- |" + " --- |" * len(s["budgets"])]
        lines += [f"| {m} | " + " | ".join(_cell(res[str(b)][m][key]) for b in s["budgets"]) + " |" for m in METHODS]
        lines += [""]
    lines += ["## k-sensitivity (mean coverage / precision at k=3 / 5 / 10)", "",
              "| method | B | k=3 | k=5 | k=10 |", "| --- | --- | --- | --- | --- |"]
    for m in METHODS:
        for b in s["budgets"]:
            ks = res[str(b)][m]["k_sensitivity"]
            lines.append(f"| {m} | {b} | " + " | ".join(f"{ks[k]['coverage']:.3f} / {ks[k]['precision']:.3f}"
                                                         for k in ("3", "5", "10")) + " |")
    lines += ["", "## Chosen lambdas (count of seeds) and Flashpoint share of the crowd", "",
              "| method | B | " + " | ".join(f"λ={x}" for x in s["lambdas"]) + " | FP share |",
              "| --- | --- |" + " --- |" * (len(s["lambdas"]) + 1)]
    for m in ("select_claude", "select_claude_plus_fp"):
        for b in s["budgets"]:
            e = res[str(b)][m]
            lines.append(f"| {m} | {b} | " + " | ".join(str(e["chosen_lambdas"][str(x)]) for x in s["lambdas"])
                         + f" | {e['fp_share_of_crowd']:.3f} |")
    lines += ["", "## Paired comparison (coverage, same seeds)", "",
              "| budget | A vs B | mean diff (A - B) | sd | A wins |", "| --- | --- | --- | --- | --- |"]
    lines += [f"| {d['budget']} | {d['a']} vs {d['b']} | {d['mean_diff']:+.3f} | {d['sd']:.3f} | "
              f"{d['wins']}/{d['n_seeds']} |" for d in r["paired"]]
    lines += ["", "## References (n = 334, 20 items, from bakeoff1; selection rows use the FULL pool as filter)", ""]
    lines += [f"- {n}: coverage {v['coverage']:.3f}, precision {v['precision']:.3f}"
              for n, v in r["references_n334_20items"].items()]
    return "\n".join(lines) + "\n"


def main() -> None:
    run()
    print((RESULTS / "h4d.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
