"""H8: export REAL results for the UI to results/ui/ (formats in docs/UI_DATA_CONTRACT.md).

`python -m blindspot.eval.export_ui`
Zero cost: no Claude or Flashpoint calls, and the holdout split is never loaded for scoring (holdout rows are
dropped at load). `results/final_holdout.json` is only READ, to add "holdout_coverage" in summary.json.
Everything in the UI is TEST-split numbers. Item sets: Twin 30 items (n=617); GSS the 20 live items (n=334),
except where a row says otherwise in its note (H4/H4b archetype rows and the fairness audit use all 24 items).
Maps: one fixed layout per dataset (UMAP on family-weighted Gower over the real test people, seed 42, scaled to
[0, 1]); a synthetic person sits at the mean xy of their 5 nearest real people. No respondent ids anywhere.
"""
import json
import time
from pathlib import Path

import numpy as np

from blindspot.coverage import evaluate
from blindspot.generators import claude_personas as cp
from blindspot.generators.baselines import gaussian_copula, independent_marginals, other_real_people
from blindspot.generators.claude_client import ResponseCache, Usage
from blindspot.generators.flashpoint_panel import load as load_fp
from blindspot.space import encode, gower
from blindspot.space.maps import _umap_xy

from .experiment0 import K, SEED, _assert_no_holdout, build_crowds, load_inputs
from .experiment_fp import FP_DIR, RUNS, restrict

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
UI = RESULTS / "ui"
NEAREST_REAL = 5

LABELS = {"retest": "Same people, retested", "other_real_people": "Other real people",
          "gaussian_copula": "Statistics (copula)", "independent_marginals": "Shuffled answers",
          "flashpoint_panel": "Flashpoint synthetic panel", "claude_mode": "Claude, typical answers",
          "claude_sampled": "Claude, random answers", "gpt41mini_twin": "GPT-4.1-mini twins"}
TWIN_ORDER = ("retest", "other_real_people", "gaussian_copula", "independent_marginals", "claude_mode",
              "gpt41mini_twin", "claude_sampled")
GSS_ORDER = ("other_real_people", "gaussian_copula", "independent_marginals", "flashpoint_panel", "claude_mode",
             "claude_sampled")
DIM_LABEL = {"age_band": "Age", "gender": "Gender", "education": "Education", "income_band": "Household income",
             "region": "Region"}
LEVEL_LABEL = {"male": "Men", "female": "Women", "high school": "High school", "less than high school":
               "Less than high school", "bachelor's": "Bachelor's degree", "graduate": "Graduate degree",
               "associate/junior college": "Associate degree", "Less than $30,000": "Under $30k",
               "$30,000-$49,999": "$30k to $50k", "$50,000-$74,999": "$50k to $75k",
               "$75,000-$109,999": "$75k to $110k", "$110,000 or more": "$110k and over"}


def r3(x) -> float:
    return round(float(x), 3)


def read(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def write(name: str, payload: dict) -> None:
    UI.mkdir(parents=True, exist_ok=True)
    (UI / f"{name}.json").write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


# ---------- crowds ----------
def twin_world() -> dict:
    from dotenv import dotenv_values

    d = next(x for x in load_inputs() if x.name == "twin2k")
    _assert_no_holdout(d.real.answers.index, d.splits, "twin real data")
    target, pool = d.real.subset(d.splits["test"]), d.real.subset(d.splits["pool"])
    crowds = build_crowds(d, target, pool, SEED)
    plan = cp.plan("twin2k", target.demographics, target.items, dotenv_values(ROOT / ".env")["CLAUDE_MODEL"])
    claude = cp.execute(plan, None, ResponseCache(), Usage(batch=True), seed=SEED, batch=True)
    crowds.update({"claude_mode": claude.mode, "claude_sampled": claude.sampled})
    return {"target": target, "crowds": crowds}


def gss_world() -> dict:
    from dotenv import dotenv_values

    d = next(x for x in load_inputs() if x.name == "gss")
    _assert_no_holdout(d.real.answers.index, d.splits, "gss real data")
    live = [i.code for i in d.real.items if i.live_ok]
    full_target = d.real.subset(d.splits["test"])
    target, pool = restrict(full_target, live), restrict(d.real.subset(d.splits["pool"]), live)
    plan = cp.plan("gss", full_target.demographics, full_target.items, dotenv_values(ROOT / ".env")["CLAUDE_MODEL"])
    claude = cp.execute(plan, None, ResponseCache(), Usage(batch=True), seed=SEED, batch=True)
    meta = json.loads((FP_DIR / "runs_meta.json").read_text(encoding="utf-8"))
    fp = load_fp(FP_DIR / meta["runs"][RUNS[0]]["rows_file"], target.items, RUNS[0])[0]
    crowds = {"claude_mode": restrict(claude.mode, live), "claude_sampled": restrict(claude.sampled, live),
              "gaussian_copula": gaussian_copula(pool, target.demographics, d.splits, SEED),
              "independent_marginals": independent_marginals(pool, target.demographics, d.splits, SEED),
              "other_real_people": other_real_people(pool, len(target.answers), d.splits, SEED),
              "flashpoint_panel": fp}
    return {"target": target, "crowds": crowds}


# ---------- maps ----------
def fixed_layout(target) -> np.ndarray:
    x = encode(target)
    dist = np.nan_to_num(gower(x, x, target.items), nan=1.0)
    np.fill_diagonal(dist, 0.0)
    xy = _umap_xy(dist, SEED)
    return (xy - xy.min(axis=0)) / (xy.max(axis=0) - xy.min(axis=0))


def synthetic_xy(target, crowd, xy: np.ndarray) -> np.ndarray:
    d = np.nan_to_num(gower(encode(crowd), encode(target), target.items), nan=1.0)
    return xy[np.argsort(d, axis=1)[:, :NEAREST_REAL]].mean(axis=1)


def write_map(dataset: str, name: str, world: dict, xy: np.ndarray, expected: float) -> dict:
    target, crowd = world["target"], world["crowds"][name]
    covered = evaluate(target, crowd, target.items, k=K, n_boot=2, seed=SEED, k_values=(K,))["_real_covered"]
    syn = synthetic_xy(target, crowd, xy)
    write(f"map_{dataset}_{name}", {
        "dataset": dataset, "crowd": name,
        "real": [[round(float(a), 4), round(float(b), 4), int(c)] for (a, b), c in zip(xy, covered)],
        "synthetic": [[round(float(a), 4), round(float(b), 4)] for a, b in syn]})
    return {"dataset": dataset, "crowd": name, "map_covered_share": r3(covered.mean()), "summary_coverage": r3(expected),
            "equal": r3(covered.mean()) == r3(expected)}


# ---------- summary ----------
def crowd_entry(name: str, cov, cov_ci, prec, prec_ci, holdout) -> dict:
    e = {"label": LABELS[name], "coverage": r3(cov), "coverage_ci": [r3(v) for v in cov_ci],
         "precision": r3(prec), "precision_ci": [r3(v) for v in prec_ci]}
    if holdout is not None:
        e["holdout_coverage"] = r3(holdout)
    return e


def build_summary(e1: dict, b1: dict, fp: dict, fh: dict) -> dict:
    twin = {}
    for n in TWIN_ORDER:
        c = e1["datasets"]["twin2k"]["crowds"][n]
        twin[n] = crowd_entry(n, c["coverage"], c["coverage_ci"], c["precision"], c["precision_ci"],
                              fh["twin2k"]["rows"][n]["coverage"]["mean"])
    gss, rows = {}, b1["A_gss_select_from_many_n334"]
    for n in GSS_ORDER:
        if n == "flashpoint_panel":
            c = fp["flashpoint"]["runs"]["q1"]
            gss[n] = crowd_entry(n, c["coverage"], c["coverage_ci"], c["precision"], c["precision_ci"],
                                 fh["gss"]["rows"]["flashpoint_q1_n200"]["coverage"]["mean"])
            gss[n].update({"n": 200, "items": 20, "note": (
                "Run 1 of 2. Test: run 1 {:.3f}, run 2 {:.3f}. Holdout: panel 1 {:.3f}, panel 2 {:.3f}; the two panels "
                "disagree more on holdout, so treat run-to-run stability as unproven.").format(
                    c["coverage"], fp["flashpoint"]["runs"]["q2"]["coverage"],
                    fh["gss"]["rows"]["flashpoint_q1_n200"]["coverage"]["mean"],
                    fh["gss"]["rows"]["flashpoint_q2_n200"]["coverage"]["mean"])})
            continue
        a = rows[n]
        gss[n] = crowd_entry(n, a["coverage"]["mean"], a["coverage_ci_first_seed"], a["precision"]["mean"],
                             a["precision_ci_first_seed"], fh["gss"]["rows"][n]["coverage"]["mean"])
    return {"model": e1["settings"]["model"], "datasets": {
        "twin": {"label": "Twin-2K-500", "split": "test", "n": 617, "items": 30, "crowds": twin},
        "gss": {"label": "GSS 2024", "split": "test", "n": 334, "items": 20, "crowds": gss}}}


# ---------- H4 / H4d based files ----------
def seed_range(vals: list[float]) -> tuple[float, float, float]:
    return float(np.mean(vals)), float(min(vals)), float(max(vals))


def build_repairs(b1: dict, fp: dict, h4d: dict, e2: dict, e2b: dict) -> dict:
    rows = b1["A_gss_select_from_many_n334"]
    base = rows["claude_mode"]["coverage"]["mean"]
    h4 = seed_range([s["blindspot_targeted"][3]["coverage"] for s in e2["datasets"]["gss"]["per_seed"].values()])
    h4b = seed_range([s["h4b_verify"][3]["coverage"] for s in e2b["datasets"]["gss"]["per_seed"].values()])
    h = h4d["results"]
    q1 = fp["flashpoint"]["runs"]["q1"]

    def d4(b: str, m: str) -> tuple[float, float, float]:
        c = h[b][m]["coverage"]
        return c["mean"], c["min"], c["max"]

    spec = [
        ("Random answers from Claude", "ai", None, (rows["claude_sampled"]["coverage"]["mean"],
                                                    *rows["claude_sampled"]["coverage_ci_first_seed"])),
        ("Flashpoint synthetic panel", "ai", "n=200", (q1["coverage"], *q1["coverage_ci"])),
        ("Teach Claude archetypes", "ai_filtered", "24 items, 3 seeds", h4),
        ("Teach Claude verified recipes", "ai_filtered", "24 items, 3 seeds", h4b),
        ("Pick best of 50 AI drafts", "ai_filtered", "100 real people as filter", d4("100", "select_claude")),
        ("Pick best drafts + Flashpoint", "ai_filtered", "100 real people as filter",
         d4("100", "select_claude_plus_fp")),
        ("25 real people + statistics", "real", None, d4("25", "copula_on_asked")),
        ("100 real people + statistics", "real", None, d4("100", "copula_on_asked"))]
    items = []
    for label, kind, note, (mean, lo, hi) in spec:
        it = {"label": label, "coverage": r3(mean), "lo": r3(lo), "hi": r3(hi), "kind": kind}
        if note:
            it["note"] = note
        items.append(it)
    best_ai = max((i for i in items if i["kind"] != "real"), key=lambda i: i["coverage"])
    c25, c100 = items[6]["coverage"], items[7]["coverage"]
    caption = (f"Claude alone covers {base:.0%} of real people. The best AI-based repair ({best_ai['label']}, "
               f"{best_ai['coverage']:.0%}) still trails statistics fitted on 100 real people ({c100:.0%}); statistics on "
               f"just 25 real people reach {c25:.0%}. Bars show the range over seeds (or a 95% interval for single runs); "
               "scored on GSS test people none of the methods saw.")
    return {"baseline": r3(base), "items": items, "caption": caption}


def build_scoreboard(b1: dict, h4d: dict) -> dict:
    h, budgets = h4d["results"], [25, 50, 75, 100]
    spec = [("copula_on_asked", "copula_on_asked", "Statistics on the real people"),
            ("claude_plus_asked", "claude_plus_asked", "AI crowd + the real people"),
            ("blindspot_select_fp", "select_claude_plus_fp", "AI drafts filtered by them (+Flashpoint)"),
            ("blindspot_select", "select_claude", "AI drafts filtered by them")]
    methods = {}
    for key, src, label in spec:
        cov = [h[str(b)][src]["coverage"] for b in budgets]
        methods[key] = {"label": label, "mean": [r3(c["mean"]) for c in cov], "lo": [r3(c["min"]) for c in cov],
                        "hi": [r3(c["max"]) for c in cov]}
    rows = b1["A_gss_select_from_many_n334"]
    refs = {"claude_mode": {"label": "AI crowd alone", "value": r3(rows["claude_mode"]["coverage"]["mean"])},
            "other_real_people": {"label": "Real people ceiling",
                                  "value": r3(rows["other_real_people"]["coverage"]["mean"])}}
    top = [max(methods, key=lambda m: methods[m]["mean"][i]) for i in range(len(budgets))]
    cop = methods["copula_on_asked"]["mean"]
    ai = max(methods["blindspot_select_fp"]["mean"][-1], methods["blindspot_select"]["mean"][-1])
    caption = (f"Statistics fitted on the real people came first at {sum(t == 'copula_on_asked' for t in top)} of "
               f"{len(budgets)} budgets ({cop[0]:.0%} with 25 people, {cop[-1]:.0%} with 100); AI drafts filtered by those "
               f"people reached at most {ai:.0%}, and the AI crowd with the people swapped in caught up only near 100 "
               f"({methods['claude_plus_asked']['mean'][-1]:.0%}). Lines show the range over 10 seeds.")
    return {"budgets": budgets, "methods": methods, "references": refs, "caption": caption}


def build_fairness(h4c: dict) -> dict:
    fa = h4c["fairness_claude_mode"]
    groups, per_dim = [], {}
    for key, v in fa["groups"].items():
        dim, level = key.split("=", 1)
        if level == "unknown":
            continue
        g = {"dimension": DIM_LABEL[dim], "label": LEVEL_LABEL.get(level, level.capitalize()), "n": v["n"],
             "coverage": r3(v["coverage"]), "lo": r3(v["ci"][0]), "hi": r3(v["ci"][1])}
        groups.append(g)
        per_dim.setdefault(dim, []).append(g)
    gaps = [(abs(a["coverage"] - b["coverage"]), a, b) for gs in per_dim.values() for i, a in enumerate(gs)
            for b in gs[i + 1:] if a["hi"] < b["lo"] or b["hi"] < a["lo"]]
    if gaps:
        _, a, b = max(gaps, key=lambda t: t[0])
        lo, hi = sorted((a, b), key=lambda g: g["coverage"])
        tail = (f"{len(gaps)} pairs of groups within a dimension have clearly separated intervals, the widest being "
                f"{lo['label'].lower()} ({lo['coverage']:.0%}) vs {hi['label'].lower()} ({hi['coverage']:.0%}); the other "
                "intervals overlap.")
    else:
        tail = "All intervals overlap, so this is a pattern, not proof."
    caption = (f"Share of each group the AI crowd can imagine (overall {fa['overall_coverage']:.0%}, all 24 items, "
               f"groups of {fa['min_group_n']}+ people, bootstrap 95% intervals). {tail}")
    return {"overall": r3(fa["overall_coverage"]), "groups": groups, "caption": caption}


def build_live(gss_w: dict, cov_claude: float) -> dict:
    target, crowd = gss_w["target"], gss_w["crowds"]["claude_mode"]

    def enc(resp) -> list:
        return [[None if np.isnan(v) else round(float(v), 4) for v in row] for row in encode(resp)]

    return {"items": [{"code": it.code, "wording": it.wording, "family": it.family,
                       "options": [o[:1].upper() + o[1:] for o in it.options]} for it in target.items],
            "family_weights": {f: 0.25 for f in sorted({it.family for it in target.items})},
            "real": enc(target), "real_xy": gss_w["xy"], "crowd": enc(crowd), "crowd_label": "Claude AI crowd",
            "base_rate_uncovered": r3(1 - cov_claude)}


# ---------- run ----------
def run() -> dict:
    started = time.perf_counter()
    e1, b1, fp, fh = read("experiment1.json"), read("bakeoff1.json"), read("experiment_fp.json"), read("final_holdout.json")
    h4d, h4c, e2, e2b = read("h4d.json"), read("h4c.json"), read("experiment2.json"), read("experiment2b.json")
    if "min" not in h4d["results"]["25"]["copula_on_asked"]["coverage"]:
        raise SystemExit("results/h4d.json lacks seed min/max: re-run `python -m blindspot.eval.h4d` first")
    summary = build_summary(e1, b1, fp, fh)
    write("summary", summary)
    checks = []
    worlds = {"twin": twin_world(), "gss": gss_world()}
    for ds, world in worlds.items():
        xy = fixed_layout(world["target"])
        world["xy"] = [[round(float(a), 4), round(float(b), 4)] for a, b in xy]
        for name, entry in summary["datasets"][ds]["crowds"].items():
            checks.append(write_map(ds, name, world, xy, entry["coverage"]))
    bad = [c for c in checks if not c["equal"]]
    if bad:
        raise SystemExit(f"coverage assertion FAILED for {bad}")
    repairs, board, fair = build_repairs(b1, fp, h4d, e2, e2b), build_scoreboard(b1, h4d), build_fairness(h4c)
    live = build_live(worlds["gss"], summary["datasets"]["gss"]["crowds"]["claude_mode"]["coverage"])
    for name, payload in (("repairs_gss", repairs), ("scoreboard_gss", board), ("fairness_gss", fair),
                          ("live_gss", live)):
        write(name, payload)
    (RESULTS / "ui_export.md").write_text(to_markdown(checks, repairs, board, fair, live, time.perf_counter() - started),
                                          encoding="utf-8")
    return {"checks": checks}


def to_markdown(checks, repairs, board, fair, live, seconds) -> str:
    lines = ["# UI export (H8)", "", f"Files in `results/ui/`, test-split numbers, runtime {seconds:.0f} s. Coverage "
             "assertion (map covered share equals summary coverage, 3 decimals):", "",
             "| dataset | crowd | map covered share | summary coverage | equal |", "| --- | --- | --- | --- | --- |"]
    lines += [f"| {c['dataset']} | {c['crowd']} | {c['map_covered_share']} | {c['summary_coverage']} | "
              f"{'yes' if c['equal'] else 'NO'} |" for c in checks]
    lines += ["", "## Captions", "", f"- repairs_gss: {repairs['caption']}", f"- scoreboard_gss: {board['caption']}",
              f"- fairness_gss: {fair['caption']}", "", f"live_gss: {len(live['items'])} items, {len(live['real'])} real "
              f"people, crowd = claude_mode, base_rate_uncovered = {live['base_rate_uncovered']}."]
    return "\n".join(lines) + "\n"


def main() -> None:
    out = run()
    print(f"wrote {len(list(UI.glob('*.json')))} files; coverage assertions passed for {len(out['checks'])} maps")
    print((RESULTS / "ui_export.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
