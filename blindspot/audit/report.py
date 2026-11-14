"""The audit report: everything the page and the downloadable HTML show, as one JSON-able dict (numpy only)."""
import numpy as np

from .augment import augment
from .engine import Estimator, Prepared, estimate, expected_error, load_calibration
from .labels import short_label, short_labels
from .layout import build_map

BUNCH_SHARE = 0.80
TOP_GAPS = 5
N_EXAMPLES = 3
DIFF_ROWS, SAME_ROWS = 8, 2
HOW_TO_READ = (
    "Each real person is a dot; people who answered alike sit close together. A real person is COVERED when someone in the "
    "AI panel is at least as close to them as their k-th most similar other real person (k-nearest-neighbour coverage, "
    "Naeem et al. 2020). MISSED = not covered. REALISTIC = share of AI respondents that fall inside some real person's "
    "neighbourhood. Distance: mean over the questions both people answered, equal weight per question; ordered answers are "
    "scaled to [0, 1], categorical answers count 0 or 1. With a small real sample the neighbourhoods are wide, so the "
    "estimate is corrected for sample size and shown with the typical error measured on a national survey (see the "
    "validation table).")


def _shares(col: np.ndarray, bins: int = 10) -> dict[float, float]:
    col = col[~np.isnan(col)]
    if not len(col):
        return {}
    vals, counts = np.unique(col, return_counts=True)
    if len(vals) > 12:                                         # many numeric values: compare in bins
        idx = np.minimum((col * bins).astype(int), bins - 1)
        vals, counts = np.unique(idx, return_counts=True)
        vals = (vals + 0.5) / bins
    return dict(zip(vals.tolist(), (counts / counts.sum()).tolist()))


def _label(spec, code: float) -> str:
    return spec.decode(code)


def question_tables(prep: Prepared, lab: dict[str, str]) -> tuple[dict, list[dict]]:
    flagged, gaps = [], []
    n_ai_bunched = n_real_bunched = 0
    for j, spec in enumerate(prep.columns):
        ai, real = _shares(prep.X_syn[:, j]), _shares(prep.X_real[:, j])
        if not ai or not real:
            continue
        ai_top, real_top = max(ai, key=ai.get), max(real, key=real.get)
        n_ai_bunched += ai[ai_top] > BUNCH_SHARE
        n_real_bunched += real[real_top] > BUNCH_SHARE
        if ai[ai_top] > BUNCH_SHARE:
            flagged.append({"question": lab[spec.name], "ai_top_answer": _label(spec, ai_top), "ai_top_share": ai[ai_top],
                            "real_top_share": real.get(ai_top, 0.0)})
        keys = sorted(set(ai) | set(real))
        diffs = {k: ai.get(k, 0.0) - real.get(k, 0.0) for k in keys}
        worst = max(diffs, key=lambda k: abs(diffs[k]))
        gaps.append({"question": lab[spec.name], "tvd": 0.5 * sum(abs(v) for v in diffs.values()),
                     "answer": _label(spec, worst), "ai_share": ai.get(worst, 0.0), "real_share": real.get(worst, 0.0)})
    flagged.sort(key=lambda r: -r["ai_top_share"])
    gaps.sort(key=lambda r: -r["tvd"])
    bunching = {"threshold": BUNCH_SHARE, "n_questions": len(prep.columns), "n_ai_bunched": int(n_ai_bunched),
                "n_real_bunched": int(n_real_bunched), "questions": flagged[:12]}
    return bunching, gaps[:TOP_GAPS]


def examples(prep: Prepared, e: dict, lab: dict[str, str]) -> list[dict]:
    """Missed real people (moderately far from the crowd, mostly answered) beside the closest AI person."""
    covered, d_rs, radius = e["covered"], e["d_rs"], e["radius"]
    nn_idx, nn = d_rs.argmin(axis=1), d_rs.min(axis=1)
    answered = (~np.isnan(prep.X_real)).mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(radius > 0, nn / radius, np.inf)
    cand = np.where(~covered & (answered >= 0.5) & np.isfinite(ratio))[0]
    if not len(cand):
        return []
    cand = cand[np.argsort(ratio[cand], kind="stable")]
    picks = []
    for q in (0.25, 0.55, 0.85)[:N_EXAMPLES]:
        i = int(cand[min(len(cand) - 1, int(q * len(cand)))])
        if i not in picks:
            picks.append(i)
    out = []
    for label, i in zip("ABC", picks):
        a, b = prep.X_real[i], prep.X_syn[nn_idx[i]]
        both = ~np.isnan(a) & ~np.isnan(b)
        diff = np.where(both, np.where(prep.cat, (a != b).astype(float), np.abs(a - b)), -1.0)
        order = [j for j in np.argsort(-diff, kind="stable") if diff[j] > 0][:DIFF_ROWS]
        same = [j for j in range(len(a)) if both[j] and diff[j] == 0][:SAME_ROWS]
        out.append({"label": f"Person {label}", "n_diff": int((diff > 0).sum()), "n_questions": int(both.sum()),
                    "rows": [{"q": lab[prep.columns[j].name], "real": prep.raw_real[i][j],
                              "ai": prep.raw_syn[nn_idx[i]][j]} for j in order + same]})
    return out


def recommendation(missed: float, realistic: float, err: dict, bunching: dict, gaps: list[dict]) -> dict:
    level = "high" if missed >= 0.5 else "medium" if missed >= 0.25 else "low"
    head = f"Your AI panel misses about {missed:.0%} of the real people in your sample."
    advice = {
        "high": "Do not publish results from this panel alone: more than half of real respondents have nobody like them in it. "
                "Blend in your real sample or collect more real responses before drawing conclusions.",
        "medium": "Use this panel with care: a sizeable minority of real respondents have nobody like them in it. "
                  "Blend in real responses for the questions below, and report results with this gap in mind.",
        "low": "The panel covers most of your real sample. Keep checking it as you add questions, and look at the "
               "questions below where its answers still differ."}[level]
    paras = [advice]
    if bunching["n_ai_bunched"]:
        qs = ", ".join(f"“{r['question']}”" for r in bunching["questions"][:3])
        paras.append(f"On {bunching['n_ai_bunched']} of {bunching['n_questions']} questions more than "
                     f"{int(BUNCH_SHARE * 100)}% of the AI panel give the same answer, which real people rarely do "
                     f"(for example {qs}). That makes the panel look more alike than people are.")
    if gaps:
        paras.append("Collect more real responses on: " + ", ".join(f"“{g['question']}”" for g in gaps[:3]) +
                     ". These are the questions where the AI panel's answers differ most from your real sample.")
    if realistic < 0.7:
        paras.append(f"About {1 - realistic:.0%} of the AI respondents do not look like any real person in your sample; "
                     "consider filtering them out.")
    if not err.get("reliable"):
        paras.append(f"Your real sample ({err['n']} people) is below the reliable minimum of {err['min_reliable_n']}: "
                     "treat the numbers as a rough guide and add more real respondents if you can.")
    return {"level": level, "headline": head, "paragraphs": paras}


def build_report(prep: Prepared, cal: dict | None = None, seed: int = 42) -> dict:
    cal = cal or load_calibration()
    est = Estimator(**cal["estimator"])
    e = estimate(prep, est, cal)
    n = len(prep.X_real)
    err = expected_error(n, cal)
    ptable = {int(k): v for k, v in cal.get("precision_error_table", {}).items()}
    realistic_error = ptable[min(ptable, key=lambda m: abs(m - n))]["rmse"] if ptable else None
    missed = 1.0 - e["coverage"]
    lab = short_labels([c.name for c in prep.columns] + [x["name"] for x in prep.entries if x["name"] not in {c.name for c in prep.columns}])
    bunching, gaps = question_tables(prep, lab)
    used = {c.name for c in prep.columns}
    return {
        "summary": {"n_real": n, "n_real_file": prep.n_real_file, "n_ai": prep.n_syn, "n_questions": len(prep.columns),
                    "missed": missed, "covered": e["coverage"], "realistic": e["precision"],
                    "estimator": {**cal["estimator"], "n_ref": cal["n_ref"], "k_ref": cal["k_ref"]},
                    "expected_error": err, "realistic_error": realistic_error},
        "columns": {
            "used": [{"name": c.name, "label": lab[c.name], "kind": c.kind, "n_options": c.n_options} for c in prep.columns],
            "ignored": [{"name": x["name"], "label": lab[x["name"]], "reason": x["reason"] or "not selected"} for x in prep.entries
                        if x["name"] not in used],
            "fuzzy": [{"panel": short_label(x["panel_header"]), "real": lab[x["name"]]} for x in prep.entries if x["method"] == "fuzzy"
                      and x["name"] in used],
            "unmatched_panel": prep.unmatched_panel, "unmatched_real": prep.unmatched_real},
        "bunching": bunching, "gaps": gaps, "examples": examples(prep, e, lab),
        "map": build_map(e["d_rr"], e["d_rs"], e["covered"], seed),
        "recommendation": recommendation(missed, e["precision"], err, bunching, gaps),
        "notes": prep.notes, "how_to_read": HOW_TO_READ,
        "repaired_note": "The repaired crowd contains your AI panel plus extra respondents generated by a statistical model "
                         "fitted on your real sample (no real respondent's row is copied). It includes information from "
                         "real respondents; do not describe it as purely AI-generated.",
    }


def repaired_rows(prep: Prepared, headers_only: bool = False, seed: int = 42) -> tuple[list[str], list[list[str]]]:
    """Header and rows of the repaired crowd: the AI panel (as uploaded) plus statistical augmentation, labelled."""
    header = [c.name for c in prep.columns] + ["source"]
    if headers_only:
        return header, []
    rows = [r + ["AI panel"] for r in prep.raw_syn]
    rows += [r + ["statistical augmentation (fitted on your real sample)"] for r in augment(prep, min(prep.n_syn, 5000), seed)]
    return header, rows
