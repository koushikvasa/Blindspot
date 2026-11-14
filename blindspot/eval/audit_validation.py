"""Small-sample validation of the audit estimator (zero cost, no API calls, the holdout is never loaded).

`python -m blindspot.eval.audit_validation`
GSS, 20 live items. Real samples of 25/50/100/200 people are drawn from the POOL split (20 draws each); AI crowds =
claude_mode, claude_sampled (cached Haiku 5.5 responses, test personas) and Flashpoint panel 1. For each (draw, crowd) the
audit engine's estimate is compared with the TRUE coverage of the same crowd: the same engine run on the whole TEST split
(334 real people, 5th-nearest-neighbour radius). Estimators tried: k = 1, 3, 5 nearest other real people, plain or
rescaled to what a 334-person real sample would give. The best (lowest pooled RMSE) is written to
blindspot/audit/calibration.json with the error table the report shows; the full tables go to results/audit_validation.md.
"""
import json
import time
from pathlib import Path

import numpy as np

from blindspot.audit.engine import (CALIBRATION, Estimator, coverage_precision, gower, prepare, radii, with_self_excluded)
from blindspot.audit.parse import Table
from blindspot.data import gss
from blindspot.data.schema import Respondents
from blindspot.generators import claude_personas as cp
from blindspot.generators.claude_client import ResponseCache, Usage
from blindspot.generators.flashpoint_panel import load as load_flashpoint

from .experiment0 import SEED, _git_commit
from .experiment_fp import FP_DIR, RUNS
from .splits import load_splits

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
SIZES = (25, 50, 100, 200)
DRAWS = 20
ESTIMATORS = [Estimator(k, False) for k in (1, 3, 5)] + [Estimator(k, True, p) for p in (0.5, 0.75, 1.0) for k in (1, 3, 5)]
N_REF, K_REF = 334, 5
RELIABLE_RMSE = 0.07            # the smallest sample size with a typical error of at most 7 points is "reliable"


def to_table(name: str, resp: Respondents, rows: list[int] | None = None) -> Table:
    items = resp.items
    idx = range(len(resp.answers)) if rows is None else rows
    a = resp.answers.to_numpy()
    body = tuple(tuple("" if np.isnan(a[i, j]) else items[j].options[int(a[i, j])] for j in range(len(items))) for i in idx)
    return Table(name, tuple(it.code for it in items), body)


def estimates(panel: Table, real: Table) -> dict[Estimator, tuple[float, float]]:
    prep = prepare(panel, real)
    d_rs = gower(prep.X_real, prep.X_syn, prep.cat)
    d_rr = with_self_excluded(gower(prep.X_real, prep.X_real, prep.cat))
    out = {}
    for est in ESTIMATORS:
        cov, prec, _ = coverage_precision(d_rs, radii(d_rr, est, N_REF, K_REF))
        out[est] = (cov, prec)
    return out


def run() -> dict:
    from dotenv import dotenv_values

    started = time.perf_counter()
    g = gss.load()
    splits = load_splits("gss")
    live = [i.code for i in g.items if i.live_ok]
    pool_full, test_full = g.subset(splits["pool"]), g.subset(splits["test"])        # holdout never read
    keep = [list(g.answers.columns).index(c) for c in live]
    items = [g.items[j] for j in keep]

    def live_only(resp: Respondents) -> Respondents:
        return Respondents(resp.answers[live], resp.demographics, items, resp.source)

    pool, test = live_only(pool_full), live_only(test_full)
    plan = cp.plan("gss", test_full.demographics, test_full.items, dotenv_values(ROOT / ".env")["CLAUDE_MODEL"])
    claude = cp.execute(plan, None, ResponseCache(), Usage(batch=True), seed=SEED, batch=True)        # cache only
    meta = json.loads((FP_DIR / "runs_meta.json").read_text(encoding="utf-8"))
    crowds = {"claude_mode": live_only(claude.mode), "claude_sampled": live_only(claude.sampled),
              "flashpoint_panel_1": load_flashpoint(FP_DIR / meta["runs"][RUNS[0]]["rows_file"], items, RUNS[0])[0]}
    tables = {k: to_table(k, v) for k, v in crowds.items()}
    test_table = to_table("test", test)
    truth = {k: estimates(t, test_table)[Estimator(K_REF, False)] for k, t in tables.items()}
    errors: dict[tuple, list] = {}                                                  # (estimator, n, crowd, metric) -> errors
    for n in SIZES:
        for draw in range(DRAWS):
            rows = sorted(np.random.default_rng([SEED, n, draw]).choice(len(pool.answers), n, replace=False).tolist())
            real = to_table("real", pool, rows)
            for crowd, panel in tables.items():
                for est, (cov, prec) in estimates(panel, real).items():
                    errors.setdefault((est, n, crowd, "coverage"), []).append(cov - truth[crowd][0])
                    errors.setdefault((est, n, crowd, "precision"), []).append(prec - truth[crowd][1])
    return {"truth": truth, "errors": errors, "seconds": time.perf_counter() - started}


def stats(values: list[float]) -> dict:
    v = np.asarray(values)
    return {"bias": float(v.mean()), "sd": float(v.std(ddof=1)), "rmse": float(np.sqrt((v ** 2).mean()))}


def name(e: Estimator) -> str:
    return f"k={e.k}{f', rescaled x{e.power:g}' if e.corrected else ''}"


def pooled(errors: dict, est: Estimator, n: int, metric: str) -> dict:
    vals = [x for (e, m, _, mt), xs in errors.items() if e == est and m == n and mt == metric for x in xs]
    return stats(vals)


def choose(errors: dict, metric: str = "coverage") -> Estimator:
    score = {e: np.mean([pooled(errors, e, n, metric)["rmse"] for n in SIZES]) for e in ESTIMATORS}
    return min(score, key=score.get)


def to_markdown(res: dict, best: Estimator, table: dict, min_reliable: int, commit: str, best_p: Estimator, ptable: dict) -> str:
    err, truth = res["errors"], res["truth"]
    lines = ["# Audit estimator: small-sample validation", "",
             f"GSS 2024, 20 live items. Real samples of {', '.join(map(str, SIZES))} people drawn from the POOL split ({DRAWS} "
             "draws each); AI crowds = Haiku 5.5 typical answers (claude_mode), Haiku 5.5 random answers (claude_sampled) and "
             "Flashpoint panel 1. The tool's estimate is compared with the TRUE coverage of the same crowd: the same engine on the "
             f"whole TEST split (334 real people, 5th-nearest-neighbour radius). Error = estimate - truth (coverage = share of "
             f"real people covered; missed = 1 - coverage, so a positive coverage bias means the tool is too optimistic). "
             f"Zero API cost, holdout never loaded. Commit `{commit}`. Runtime {res['seconds']:.0f} s.", "",
             "True coverage on the test split: " + ", ".join(f"{k} {v[0]:.3f} (precision {v[1]:.3f})" for k, v in truth.items()), "",
             "## Estimators (pooled over the three crowds; coverage error in points = 100 x error)", "",
             "| estimator | " + " | ".join(f"n={n}: bias / sd / rmse" for n in SIZES) + " | mean rmse |", "| --- |" + " --- |" * (len(SIZES) + 1)]
    for e in ESTIMATORS:
        cells = [pooled(err, e, n, "coverage") for n in SIZES]
        lines.append(f"| {name(e)}{' **(chosen)**' if e == best else ''} | " + " | ".join(
            f"{100 * c['bias']:+.1f} / {100 * c['sd']:.1f} / {100 * c['rmse']:.1f}" for c in cells)
            + f" | {100 * np.mean([c['rmse'] for c in cells]):.1f} |")
    lines += ["", f"## Chosen estimator: {name(best)}", "", "Per crowd (coverage error in points: bias / sd / rmse):", "",
              "| crowd | " + " | ".join(f"n={n}" for n in SIZES) + " |", "| --- |" + " --- |" * len(SIZES)]
    for crowd in truth:
        cells = [stats(err[(best, n, crowd, "coverage")]) for n in SIZES]
        lines.append(f"| {crowd} | " + " | ".join(f"{100 * c['bias']:+.1f} / {100 * c['sd']:.1f} / {100 * c['rmse']:.1f}" for c in cells) + " |")
    lines += ["", "## Precision (share of AI respondents that look realistic)", "",
              "Precision depends on how many real people the AI cloud can be compared with, so small samples understate it. "
              "Pooled error in points (bias / sd / rmse):", "", "| estimator | " + " | ".join(f"n={n}" for n in SIZES) + " | mean rmse |",
              "| --- |" + " --- |" * (len(SIZES) + 1)]
    for e in ESTIMATORS:
        cells = [pooled(err, e, n, "precision") for n in SIZES]
        lines.append(f"| {name(e)}{' **(chosen for precision)**' if e == best_p else ''} | " + " | ".join(
            f"{100 * c['bias']:+.1f} / {100 * c['sd']:.1f} / {100 * c['rmse']:.1f}" for c in cells)
            + f" | {100 * np.mean([c['rmse'] for c in cells]):.1f} |")
    lines += ["",
        "## Table shipped with the report (`blindspot/audit/calibration.json`)", "", "| real sample size | bias | sd | typical error (rmse) |",
        "| --- | --- | --- | --- |"]
    lines += [f"| {n} | {100 * r['bias']:+.1f} pts | {100 * r['sd']:.1f} pts | ±{100 * r['rmse']:.1f} pts |" for n, r in table.items()]
    lines += ["", "Precision error table shipped with the report: " + ", ".join(
        f"n={n}: {100 * r['bias']:+.1f} / ±{100 * r['rmse']:.1f} pts" for n, r in ptable.items()), "",
              f"Reliable minimum: **{min_reliable} real people** (the smallest size whose typical coverage error is at most "
              f"{100 * RELIABLE_RMSE:.0f} points). Below it the report warns that the numbers are only a rough guide."]
    return "\n".join(lines) + "\n"


def main() -> None:
    res = run()
    best = choose(res["errors"])
    table = {n: pooled(res["errors"], best, n, "coverage") for n in SIZES}
    reliable = [n for n in SIZES if table[n]["rmse"] <= RELIABLE_RMSE]
    min_reliable = min(reliable) if reliable else max(SIZES)
    best_p = choose(res["errors"], "precision")
    ptable = {n: pooled(res["errors"], best_p, n, "precision") for n in SIZES}
    CALIBRATION.write_text(json.dumps({"estimator": {"k": best.k, "corrected": best.corrected, "power": best.power}, "n_ref": N_REF, "k_ref": K_REF,
                                       "min_reliable_n": min_reliable,
                                       "precision_estimator": {"k": best_p.k, "corrected": best_p.corrected, "power": best_p.power},
                                       "precision_error_table": {str(n): {k: round(v, 4) for k, v in r.items()} for n, r in ptable.items()},
                                       "error_table": {str(n): {k: round(v, 4) for k, v in r.items()} for n, r in table.items()},
                                       "validated_on": "GSS 2024, 20 live items, test split truth, pool samples, 20 draws each"},
                                      indent=1), encoding="utf-8")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "audit_validation.md").write_text(to_markdown(res, best, table, min_reliable, _git_commit(), best_p, ptable), encoding="utf-8")
    print((RESULTS / "audit_validation.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
