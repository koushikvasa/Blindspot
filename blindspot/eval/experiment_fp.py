"""Experiment FP (H5): audit Flashpoint's synthetic panel with our metrics. No network, no payments.

`python -m blindspot.eval.experiment_fp`
Target = GSS TEST split on the 20 live items (live_ok). Holdout is dropped right after loading.
Flashpoint crowds: two separate n=200 panels (uncalibrated), loaded from data/raw/flashpoint/ (git-ignored).
Fair comparison at n=200: claude_mode (cached H3 answers, subsampled from the 334-person crowd),
gaussian_copula and other_real_people (fit/drawn on pool, fresh n=200 per seed) and independent_marginals,
each over 20 seeded draws (mean, sd, 2.5-97.5 percentile ACROSS seeds, which is not a bootstrap CI).
Writes results/experiment_fp.json and .md. No Flashpoint ids or raw rows are written.
"""
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from blindspot.coverage import evaluate, family_entropy, real_distances, subsample
from blindspot.data import gss
from blindspot.data.schema import Respondents
from blindspot.generators import claude_personas as cp
from blindspot.generators.baselines import gaussian_copula, independent_marginals, other_real_people
from blindspot.generators.claude_client import ResponseCache, Usage
from blindspot.generators.flashpoint_panel import SOURCE, load as load_flashpoint

from .experiment0 import K, K_VALUES, N_BOOT, SEED, _assert_no_holdout, _fmt_ci, _git_commit
from .experiment1 import bunching_share
from .splits import load_splits

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FP_DIR = ROOT / "data/raw/flashpoint"
N_PANEL = 200
N_SEEDS = 20
SEED_BOOT = 20          # bootstrap resamples inside the per-seed baseline evaluations (CIs there are unused)
RUNS = ("q1", "q2")


def restrict(resp: Respondents, codes: list[str]) -> Respondents:
    """Same people, only the given item columns (order of `codes`)."""
    items = {i.code: i for i in resp.items}
    return Respondents(resp.answers[codes], resp.demographics, [items[c] for c in codes], resp.source,
                       dict(resp.meta))


def tv_distance(crowd: Respondents, real: Respondents) -> float:
    """Mean over items of the total variation distance between answer distributions (missing ignored)."""
    out = []
    for it in real.items:
        k = len(it.options)
        p = crowd.answers[it.code].dropna().astype(int).value_counts(normalize=True).reindex(range(k), fill_value=0)
        q = real.answers[it.code].dropna().astype(int).value_counts(normalize=True).reindex(range(k), fill_value=0)
        out.append(0.5 * float(np.abs(p - q).sum()))
    return float(np.mean(out))


def score(target: Respondents, crowd: Respondents, rr: np.ndarray, n_boot: int, seed: int) -> dict:
    m = evaluate(target, crowd, target.items, k=K, n_boot=n_boot, seed=seed, k_values=K_VALUES, real_dist=rr)
    m = {k: v for k, v in m.items() if not k.startswith("_") and k != "uncovered_ids"}
    m["bunching_gt_80"] = bunching_share(crowd)
    m["marginal_tv"] = tv_distance(crowd, target)
    return m


def claude_mode_full(target_full: Respondents, codes: list[str]) -> Respondents:
    """Budget-0 claude_mode crowd for the test split from the H3 response cache (no API calls)."""
    from dotenv import dotenv_values

    model = dotenv_values(ROOT / ".env")["CLAUDE_MODEL"]
    plan = cp.plan("gss", target_full.demographics, target_full.items, model)
    crowds = cp.execute(plan, None, ResponseCache(), Usage(batch=True), seed=SEED, batch=True)
    out = restrict(crowds.mode, codes)      # H3 left unparseable answers as NaN (44 people); Gower skips them
    out.meta["people_with_missing_items_h3"] = len(crowds.failures)
    out.meta["missing_answers_live_items"] = int(out.answers.isna().sum().sum())
    return out


def baseline_crowds(seed: int, target: Respondents, pool: Respondents, splits: dict, claude_full: Respondents) -> dict:
    demo = target.demographics.sample(N_PANEL, random_state=seed)
    return {
        "claude_mode": subsample(claude_full, N_PANEL, seed),
        "gaussian_copula": gaussian_copula(pool, demo, splits, seed),
        "other_real_people": other_real_people(pool, N_PANEL, splits, seed),
        "independent_marginals": independent_marginals(pool, demo, splits, seed),
    }


def _agg(values: list[float]) -> dict:
    v = np.asarray(values, float)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
            "p2_5": float(np.percentile(v, 2.5)), "p97_5": float(np.percentile(v, 97.5)),
            "min": float(v.min()), "max": float(v.max())}


def summarize(per_seed: list[dict]) -> dict:
    """Mean / sd / 2.5-97.5 percentile across seeds for every scalar metric."""
    out = {k: _agg([s[k] for s in per_seed]) for k in ("coverage", "precision", "rcg_ratio", "bunching_gt_80",
                                                        "marginal_tv")}
    out["k_sensitivity"] = {kk: {"coverage": _agg([s["k_sensitivity"][kk]["coverage"] for s in per_seed]),
                                 "precision": _agg([s["k_sensitivity"][kk]["precision"] for s in per_seed])}
                            for kk in per_seed[0]["k_sensitivity"]}
    fams = per_seed[0]["per_family_entropy"]["crowd"]
    out["per_family_entropy"] = {f: float(np.mean([s["per_family_entropy"]["crowd"][f] for s in per_seed]))
                                 for f in fams}
    out["n_synthetic"] = per_seed[0]["n_synthetic"]
    out["n_seeds"] = len(per_seed)
    return out


def run(out_dir: Path = RESULTS) -> dict:
    g = gss.load()
    splits = load_splits("gss")
    live = [i.code for i in g.items if i.live_ok]
    g = g.subset(splits["pool"] + splits["test"])               # holdout dropped here, never used again
    _assert_no_holdout(g.answers.index, splits, "gss real data")
    full_target, full_pool = g.subset(splits["test"]), g.subset(splits["pool"])
    target, pool = restrict(full_target, live), restrict(full_pool, live)
    rr = real_distances(target, target.items)
    meta = json.loads((FP_DIR / "runs_meta.json").read_text(encoding="utf-8"))

    fp, mapping, fp_scores = {}, {}, {}
    for tag in RUNS:
        fp[tag], rep = load_flashpoint(FP_DIR / meta["runs"][tag]["rows_file"], target.items, tag)
        mapping[tag] = rep.to_dict()
        fp_scores[tag] = score(target, fp[tag], rr, N_BOOT, SEED)   # keeps the 200-resample bootstrap CIs

    claude_full = claude_mode_full(full_target, live)
    per_seed: dict[str, list[dict]] = {}
    for i in range(N_SEEDS):
        seed = SEED + i
        for name, crowd in baseline_crowds(seed, target, pool, splits, claude_full).items():
            per_seed.setdefault(name, []).append(score(target, crowd, rr, SEED_BOOT, seed))
    times = {t: [meta["runs"][t]["run_started"], meta["runs"][t]["run_completed"]] for t in RUNS}
    report = {
        "settings": {"k": K, "k_values": list(K_VALUES), "n_boot_flashpoint": N_BOOT, "seed": SEED,
                     "n_panel": N_PANEL, "n_seeds": N_SEEDS, "items": live, "n_items": len(live),
                     "target_split": "test", "n_target": len(target.answers), "n_pool": len(pool.answers),
                     "flashpoint_source": SOURCE, "flashpoint_model": meta["model"],
                     "flashpoint_calibration": meta["calibration_version"],
                     "flashpoint_run_times_utc": times, "flashpoint_panel_spec": meta["panel_spec"],
                     "claude_mode_note": "H3 answers generated with all 24 items in one prompt; the 4 sensitive "
                                         "items are dropped afterwards. Subsampled to 200 from the 334 test personas. "
                                         f"{claude_full.meta['missing_answers_live_items']} answers on the live items "
                                         "stay NaN as in H3 (unparseable after retry).",
                     "git_commit": _git_commit(), "python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__},
        "real_target": {"family_entropy": family_entropy(target), "bunching_gt_80": bunching_share(target)},
        "flashpoint": {"runs": fp_scores, "mapping": mapping,
                       "run_to_run": {"marginal_tv_between_runs": tv_distance(fp["q1"], fp["q2"])}},
        "baselines_at_200": {name: summarize(v) for name, v in per_seed.items()},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "experiment_fp.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (out_dir / "experiment_fp.md").write_text(to_markdown(report), encoding="utf-8")
    return report


def _rng(a: dict) -> str:
    return f"{a['mean']:.3f} (sd {a['sd']:.3f}, [{a['p2_5']:.3f}, {a['p97_5']:.3f}])"


def _k_cell(m: dict, k: int, stat=lambda x: x) -> str:
    c, p = m["k_sensitivity"][str(k)]["coverage"], m["k_sensitivity"][str(k)]["precision"]
    return f"{stat(c):.3f} / {stat(p):.3f}"


def to_markdown(r: dict) -> str:
    s, fpr, base = r["settings"], r["flashpoint"]["runs"], r["baselines_at_200"]
    ks, t = s["k_values"], s["flashpoint_run_times_utc"]
    lines = [
        "# Experiment FP: Flashpoint synthetic panel vs our baselines (GSS test split)", "",
        f"Target = GSS test split (n = {s['n_target']}), {s['n_items']} live items, family-weighted Gower, "
        f"k = {s['k']}. Flashpoint: two separate n = {s['n_panel']} panels, model {s['flashpoint_model']}, "
        f"calibration `{s['flashpoint_calibration']}`, run {t['q1'][0]} to {t['q2'][1]}. Other crowds: mean over "
        f"{s['n_seeds']} seeded draws of n = {s['n_panel']} (range = 2.5-97.5 percentile across seeds, not a "
        f"bootstrap CI). Commit `{s['git_commit']}`.", "",
        "## Main table (n = 200 crowds)", "",
        "| crowd | coverage | precision | rcg_ratio | items >80% on one option | mean per-item TV vs real |",
        "| --- | --- | --- | --- | --- | --- |"]
    for tag in RUNS:
        m = fpr[tag]
        lines.append(f"| **flashpoint_{tag}** | {_fmt_ci(m['coverage'], m['coverage_ci'])} | "
                     f"{_fmt_ci(m['precision'], m['precision_ci'])} | {m['rcg_ratio']:.3f} | "
                     f"{m['bunching_gt_80']:.2f} | {m['marginal_tv']:.3f} |")
    for name, a in sorted(base.items(), key=lambda kv: -kv[1]["coverage"]["mean"]):
        lines.append(f"| {name} | {_rng(a['coverage'])} | {_rng(a['precision'])} | {a['rcg_ratio']['mean']:.3f} | "
                     f"{a['bunching_gt_80']['mean']:.2f} | {a['marginal_tv']['mean']:.3f} |")
    lines += ["", f"Real target: {r['real_target']['bunching_gt_80']:.2f} of items have >80% on one option.", "",
              "## k-sensitivity (coverage / precision)", "",
              "| crowd | " + " | ".join(f"k={k}" for k in ks) + " |", "| --- |" + " --- |" * len(ks)]
    lines += [f"| **flashpoint_{tag}** | " + " | ".join(_k_cell(fpr[tag], k) for k in ks) + " |" for tag in RUNS]
    lines += [f"| {n} | " + " | ".join(_k_cell(a, k, lambda x: x["mean"]) for k in ks) + " |"
              for n, a in base.items()]
    names = ["real target"] + [f"flashpoint_{x}" for x in RUNS] + list(base)
    lines += ["", "## Per-family mean normalized entropy", "", "| family | " + " | ".join(names) + " |",
              "| --- |" + " --- |" * len(names)]
    for fam, real_h in r["real_target"]["family_entropy"].items():
        vals = [real_h] + [fpr[x]["per_family_entropy"]["crowd"][fam] for x in RUNS] + [
            base[n]["per_family_entropy"][fam] for n in base]
        lines.append(f"| {fam} | " + " | ".join(f"{v:.2f}" for v in vals) + " |")
    rtr = r["flashpoint"]["run_to_run"]["marginal_tv_between_runs"]
    lines += ["", f"Run-to-run: coverage {fpr['q1']['coverage']:.3f} vs {fpr['q2']['coverage']:.3f}, precision "
              f"{fpr['q1']['precision']:.3f} vs {fpr['q2']['precision']:.3f}; mean per-item TV between the two "
              f"panels' answer distributions {rtr:.3f}.", "", "## Answer mapping (labels -> option indices)", ""]
    for tag in RUNS:
        mp = r["flashpoint"]["mapping"][tag]
        extra = f"; unmapped: {mp['unmapped_by_item']}" if mp["unmapped"] else ""
        lines.append(f"- {tag}: {mp['n_answers']} answers, {mp['missing']} missing, {mp['unmapped']} unmapped "
                     f"({mp['invalid_share']:.2%} invalid){extra}")
    sp = s["flashpoint_panel_spec"]
    lines += ["", "## Flashpoint panel spec", "", f"- Sampled to plan: {', '.join(sp['sampled_to_plan'])}",
              f"- Education deviation: {sp['education_deviation']}",
              f"- Invented by the generator: {', '.join(sp['invented_by_generator'])}", f"- Joint: {sp['joint']}",
              f"- Reached via: {sp['reachable_via']}", "", f"claude_mode: {s['claude_mode_note']}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    run()
    print((RESULTS / "experiment_fp.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
