"""Generate sample UI data in web/mock/ following docs/UI_DATA_CONTRACT.md.

Summary and scoreboard numbers are the real Experiment 0/1/2 results reported on 2026-11-11..13.
Maps, agent rounds and live items are SAMPLE data for layout only; the UI marks them as sample.
Run: python scripts/make_mock_ui.py
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "web" / "mock"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(7)


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, separators=(",", ":")))


def crowd(cov, cov_ci, prec, prec_ci, label):
    return {"label": label, "coverage": cov, "coverage_ci": cov_ci, "precision": prec, "precision_ci": prec_ci}


summary = {
    "model": "claude-haiku-5-5",
    "datasets": {
        "twin": {"label": "Twin-2K-500", "split": "test", "n": 617, "crowds": {
            "retest": crowd(0.984, [0.974, 0.994], 0.958, [0.940, 0.974], "Same people, retested"),
            "other_real_people": crowd(0.981, [0.968, 0.990], 0.908, [0.885, 0.930], "Other real people"),
            "gaussian_copula": crowd(0.888, [0.864, 0.909], 0.862, [0.831, 0.885], "Statistics (copula)"),
            "independent_marginals": crowd(0.512, [0.470, 0.549], 0.577, [0.535, 0.616], "Shuffled answers"),
            "claude_mode": crowd(0.345, [0.308, 0.383], 0.963, [0.948, 0.977], "Claude, typical answers"),
            "gpt41mini_twin": crowd(0.344, [0.301, 0.383], 0.956, [0.937, 0.972], "GPT-4.1-mini twins"),
            "claude_sampled": crowd(0.170, [0.141, 0.198], 0.269, [0.237, 0.305], "Claude, random answers"),
        }},
        "gss": {"label": "GSS 2024", "split": "test", "n": 334, "crowds": {
            "other_real_people": crowd(0.961, [0.940, 0.979], 0.895, [0.856, 0.928], "Other real people"),
            "gaussian_copula": crowd(0.880, [0.841, 0.916], 0.880, [0.847, 0.910], "Statistics (copula)"),
            "independent_marginals": crowd(0.740, [0.701, 0.788], 0.778, [0.722, 0.826], "Shuffled answers"),
            "claude_mode": crowd(0.638, [0.581, 0.698], 0.997, [0.988, 1.000], "Claude, typical answers"),
            "claude_sampled": crowd(0.371, [0.326, 0.413], 0.521, [0.461, 0.575], "Claude, random answers"),
        }},
    },
}
dump("summary.json", summary)

# ---------- sample maps ----------
def make_real(n, k=9):
    centers = rng.uniform(0.12, 0.88, size=(k, 2))
    sizes = rng.dirichlet(np.ones(k) * 2) * n
    sizes = np.maximum(1, np.round(sizes)).astype(int)
    sizes[-1] += n - sizes.sum()
    pts = np.vstack([c + rng.normal(0, 0.055, size=(s, 2)) for c, s in zip(centers, sizes)])
    return np.clip(pts, 0.02, 0.98), centers


def covered_flags(real, syn, k=5):
    d_rr = np.linalg.norm(real[:, None] - real[None], axis=2)
    np.fill_diagonal(d_rr, np.inf)
    radius = np.sort(d_rr, axis=1)[:, k - 1]
    if len(syn) == 0:
        return np.zeros(len(real), int)
    d_rs = np.linalg.norm(real[:, None] - syn[None], axis=2).min(axis=1)
    return (d_rs <= radius).astype(int)


def synth(real, target_cov, mode):
    """Place synthetic people so that roughly target_cov of real people are covered."""
    n = len(real)
    order = np.argsort(np.linalg.norm(real - real.mean(0), axis=1))  # central first
    if mode == "collapsed":
        keep = real[order[: int(n * target_cov)]]
        syn = keep[rng.integers(0, len(keep), n)] + rng.normal(0, 0.012, (n, 2))
    elif mode == "noisy":
        syn = rng.uniform(0.02, 0.98, (n, 2))
    else:  # spread
        idx = rng.permutation(n)[: int(n * min(1, target_cov + 0.05))]
        syn = real[idx] + rng.normal(0, 0.02, (len(idx), 2))
        syn = np.vstack([syn, real[rng.integers(0, n, n - len(idx))] + rng.normal(0, 0.03, (n - len(idx), 2))])
    return np.clip(syn, 0.01, 0.99)


def write_map(dataset, name, real, syn, target=None):
    cov = covered_flags(real, syn)
    if target is not None:  # sample data: make the dot colours match the reported coverage
        d = np.linalg.norm(real[:, None] - syn[None], axis=2).min(axis=1)
        cov = np.zeros(len(real), int)
        cov[np.argsort(d)[: int(round(target * len(real)))]] = 1
    dump(f"map_{dataset}_{name}.json", {
        "dataset": dataset, "crowd": name,
        "real": [[round(float(x), 4), round(float(y), 4), int(c)] for (x, y), c in zip(real, cov)],
        "synthetic": [[round(float(x), 4), round(float(y), 4)] for x, y in syn],
    })
    return cov.mean()


for dataset in ("twin", "gss"):
    n = summary["datasets"][dataset]["n"]
    real, _ = make_real(n)
    for key, c in summary["datasets"][dataset]["crowds"].items():
        mode = "noisy" if key in ("claude_sampled", "independent_marginals") else (
            "collapsed" if key in ("claude_mode", "gpt41mini_twin") else "spread")
        write_map(dataset, key, real, synth(real, c["coverage"], mode), c["coverage"])
    if dataset == "gss":
        gss_real = real

# ---------- sample agent rounds (GSS) ----------
gap_names = [
    ("Distrustful economic pessimists", "Low trust in institutions together with a gloomy view of the economy and their own finances."),
    ("Content but disengaged", "Satisfied with life and finances, little interest in politics, neutral on technology."),
    ("Tech-wary optimists", "Happy and socially active, but uneasy about AI, driverless cars and robot surgery."),
    ("Squeezed and sceptical", "Feels worse off than last year and doubts that technology helps people like them."),
]
rounds, covs = [], [0.638, 0.69, 0.75, 0.80, 0.84]
for r, target in enumerate(covs):
    syn = synth(gss_real, target, "collapsed" if r == 0 else "spread")
    write_map("gss", f"agent_r{r}", gss_real, syn, target)
    gaps = []
    for g, (name, summ) in enumerate(gap_names[: r]):
        status = "learned" if g in (0, 1) else ("cannot_imagine" if g == 2 else ("learned" if r >= 4 else "rejected"))
        gaps.append({"id": f"g{g + 1}", "name": name, "summary": summ, "size": int(4 + 3 * g + r),
                     "status": status, "recipes_tried": 3 if status != "cannot_imagine" else 6,
                     "recipes_accepted": 1 if status == "learned" else 0,
                     "recruited": 12 if status == "cannot_imagine" else 0,
                     "observed_rate": 0.41 if status == "cannot_imagine" else None})
    rounds.append({"round": r, "asked": 25 * r, "coverage": target, "map": f"map_gss_agent_r{r}.json", "gaps": gaps})
dump("agent_gss.json", {"method": "blindspot_verified", "seed": 42, "rounds": rounds})

# ---------- scoreboard: real H4 (Experiment 2) numbers ----------
dump("scoreboard_gss.json", {
    "budgets": [25, 50, 75, 100],
    "methods": {
        "blindspot_targeted": {"label": "Blindspot (H4)", "mean": [0.641, 0.662, 0.655, 0.656],
                                "lo": [0.638, 0.647, 0.638, 0.647], "hi": [0.647, 0.677, 0.674, 0.671]},
        "blindspot_random": {"label": "Blindspot, random asking", "mean": [0.641, 0.652, 0.648, 0.659],
                              "lo": [0.638, 0.635, 0.632, 0.635], "hi": [0.647, 0.662, 0.680, 0.671]},
        "copula_on_asked": {"label": "Statistics on asked people", "mean": [0.764, 0.848, 0.829, 0.834],
                             "lo": [0.716, 0.808, 0.808, 0.826], "hi": [0.817, 0.868, 0.841, 0.841]},
        "claude_plus_asked": {"label": "Claude + asked people", "mean": [0.721, 0.818, 0.853, 0.895],
                               "lo": [0.698, 0.814, 0.835, 0.880], "hi": [0.734, 0.823, 0.862, 0.907]},
    },
    "references": {"claude_mode": {"label": "Claude alone", "value": 0.638},
                   "other_real_people": {"label": "Real people ceiling", "value": 0.961}},
    "caption": "In the first agent design, adding the asked real people beat learned recipes at every budget.",
})

# ---------- sample live items ----------
items = [
    ("happy", "Taken all together, how would you say things are these days?", ["Very happy", "Pretty happy", "Not too happy"], "work_life"),
    ("life", "In general, do you find life exciting, pretty routine, or dull?", ["Exciting", "Routine", "Dull"], "work_life"),
    ("socfrend", "How often do you spend a social evening with friends?", ["Almost daily", "Weekly", "Monthly", "A few times a year", "Never"], "work_life"),
    ("satfin", "How satisfied are you with your present financial situation?", ["Pretty well satisfied", "More or less satisfied", "Not satisfied at all"], "money_economy"),
    ("finalter", "During the last few years, has your financial situation been getting better, worse, or stayed the same?", ["Better", "Stayed the same", "Worse"], "money_economy"),
    ("trppl", "Generally speaking, how much do you trust most people? (sample wording)", ["Not at all", "A little", "Somewhat", "A lot", "Completely"], "trust_institutions"),
    ("techesy", "Technology makes our lives easier. (sample wording)", ["Strongly disagree", "Disagree", "Neither", "Agree", "Strongly agree"], "tech_science"),
    ("aiworry", "AI will take away jobs. (sample wording)", ["Strongly disagree", "Disagree", "Neither", "Agree", "Strongly agree"], "tech_science"),
    ("aidrive", "How comfortable would you be riding in a driverless car? (sample wording)", ["Not at all", "Slightly", "Moderately", "Very", "Completely"], "tech_science"),
]
fams = sorted({f for *_, f in items})
real_vals = rng.integers(0, 5, size=(len(gss_real), len(items)))
enc = lambda row: [round(min(v, len(items[j][2]) - 1) / (len(items[j][2]) - 1), 4) for j, v in enumerate(row)]
dump("live_gss.json", {
    "items": [{"code": c, "wording": w, "options": o, "family": f} for c, w, o, f in items],
    "family_weights": {f: 1 / len(fams) for f in fams},
    "real": [enc(r) for r in real_vals],
    "real_xy": [[round(float(x), 4), round(float(y), 4)] for x, y in gss_real],
    "crowd": [enc(r) for r in rng.integers(1, 3, size=(len(gss_real), len(items)))],
    "crowd_label": "Sample crowd",
    "base_rate_uncovered": 0.36,
})
print("wrote", sorted(p.name for p in OUT.iterdir()))
