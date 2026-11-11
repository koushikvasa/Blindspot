"""Experiment 0 (H2): do synthetic crowds leave real people uncovered?

`python -m blindspot.eval.experiment0`
Target = the TEST split. Baselines are fit on the POOL split. The HOLDOUT split is dropped right
after loading and never used; every crowd and output is checked against it.
Writes results/experiment0.json, results/experiment0.md and results/maps/*.json.
"""
import json
import platform
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from blindspot.coverage import evaluate, family_entropy, real_distances
from blindspot.data.schema import Respondents
from blindspot.generators.baselines import gaussian_copula, independent_marginals, other_real_people
from blindspot.space import map_coords

from .splits import load_splits

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
K = 5
K_VALUES = (3, 5, 10)
N_BOOT = 200
SEED = 42


@dataclass(frozen=True)
class DatasetInput:
    name: str
    title: str
    real: Respondents                       # pool + test people only (holdout already removed)
    splits: dict[str, list[str]]
    fixed_crowds: dict[str, Respondents] = field(default_factory=dict)  # same people as the target
    map_crowds: tuple[str, ...] = ()


class HoldoutError(AssertionError):
    pass


def _assert_no_holdout(ids, splits: dict[str, list[str]], where: str) -> None:
    hit = set(map(str, ids)) & set(splits["holdout"])
    if hit:
        raise HoldoutError(f"{len(hit)} holdout ids reached {where}")


def build_crowds(d: DatasetInput, target: Respondents, pool: Respondents, seed: int) -> dict[str, Respondents]:
    demo = target.demographics
    crowds = dict(d.fixed_crowds)
    crowds["independent_marginals"] = independent_marginals(pool, demo, d.splits, seed)
    crowds["gaussian_copula"] = gaussian_copula(pool, demo, d.splits, seed)
    crowds["other_real_people"] = other_real_people(pool, len(target.answers), d.splits, seed)
    for name, crowd in crowds.items():
        _assert_no_holdout(crowd.answers.index, d.splits, f"crowd {name}")
    return crowds


def _public(metrics: dict) -> dict:
    return {k: v for k, v in metrics.items() if not k.startswith("_")}


def run_dataset(d: DatasetInput, out_dir: Path, k: int, k_values: tuple[int, ...], n_boot: int,
                seed: int, maps: bool) -> dict:
    started = time.perf_counter()
    _assert_no_holdout(d.real.answers.index, d.splits, f"{d.name} real data")
    target = d.real.subset(d.splits["test"])
    pool = d.real.subset(d.splits["pool"])
    crowds = build_crowds(d, target, pool, seed)
    rr = real_distances(target, target.items)
    results = {name: evaluate(target, crowd, target.items, k=k, n_boot=n_boot, seed=seed,
                              k_values=k_values, real_dist=rr)
               for name, crowd in crowds.items()}
    eval_seconds = round(time.perf_counter() - started, 1)
    map_files = {}
    if maps and d.map_crowds:
        flags = {n: {"real_covered": results[n]["_real_covered"],
                     "synthetic_inside": results[n]["_synthetic_inside"]} for n in d.map_crowds}
        paths = map_coords(target, {n: crowds[n] for n in d.map_crowds}, seed=seed,
                           covered_by_name=flags, dataset=d.name, out_dir=out_dir / "maps")
        map_files = {n: f"maps/{p.name}" for n, p in paths.items()}
    return {
        "title": d.title,
        "n_target": len(target.answers), "n_pool": len(pool.answers),
        "items": [i.code for i in target.items],
        "real_family_entropy": family_entropy(target),
        "crowds": {n: _public(m) for n, m in results.items()},
        "map_files": map_files,
        "eval_seconds": eval_seconds,
        "runtime_seconds": round(time.perf_counter() - started, 1),
    }


def _git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _fmt_ci(value: float, ci: list[float]) -> str:
    return f"{value:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def _dataset_markdown(ds: dict, k_values: list[int]) -> list[str]:
    crowds = sorted(ds["crowds"].items(), key=lambda kv: -kv[1]["coverage"])
    lines = [f"## {ds['title']} (target n = {ds['n_target']}, pool n = {ds['n_pool']})", "",
             "| crowd | n | coverage [95% CI] | precision [95% CI] | rcg_ratio |",
             "| --- | --- | --- | --- | --- |"]
    lines += [f"| {n} | {m['n_synthetic']} | {_fmt_ci(m['coverage'], m['coverage_ci'])} | "
              f"{_fmt_ci(m['precision'], m['precision_ci'])} | {m['rcg_ratio']:.3f} |" for n, m in crowds]
    lines += ["", "k-sensitivity (coverage / precision):", "",
              "| crowd | " + " | ".join(f"k={k}" for k in k_values) + " |", "| --- |" + " --- |" * len(k_values)]
    lines += [f"| {n} | " + " | ".join(f"{m['k_sensitivity'][str(k)]['coverage']:.3f} / "
                                      f"{m['k_sensitivity'][str(k)]['precision']:.3f}" for k in k_values) + " |"
              for n, m in crowds]
    names = [n for n, _ in crowds]
    lines += ["", "Per-family mean normalized entropy (0 = everyone gives the same answer, 1 = uniform):", "",
              "| family | real target | " + " | ".join(names) + " |", "| --- |" + " --- |" * (len(names) + 1)]
    for fam, real_h in ds["real_family_entropy"].items():
        lines.append(f"| {fam} | {real_h:.2f} | " + " | ".join(
            f"{ds['crowds'][n]['per_family_entropy']['crowd'][fam]:.2f}" for n in names) + " |")
    return lines + [""]


def to_markdown(report: dict) -> str:
    s = report["settings"]
    lines = ["# Experiment 0: does a synthetic crowd leave real people uncovered?", "",
             f"Target = test split; baselines fit on pool; holdout untouched. k = {s['k']}, "
             f"{s['n_boot']} bootstrap resamples (coverage: real people; precision: synthetic people; "
             f"radii fixed), seed {s['seed']}, "
             f"family-weighted Gower distance, inside = distance <= k-NN radius. "
             f"Commit `{s['git_commit']}`. Runtime {report['runtime_seconds']} s.", ""]
    for ds in report["datasets"].values():
        lines += _dataset_markdown(ds, s["k_values"])
    return "\n".join(lines)


def run(inputs: list[DatasetInput], out_dir: Path = RESULTS, k: int = K, k_values: tuple[int, ...] = K_VALUES,
        n_boot: int = N_BOOT, seed: int = SEED, maps: bool = True) -> dict:
    started = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = {d.name: run_dataset(d, out_dir, k, k_values, n_boot, seed, maps) for d in inputs}
    report = {
        "settings": {"k": k, "k_values": list(k_values), "n_boot": n_boot, "seed": seed,
                     "target_split": "test", "fit_split": "pool", "distance": "gower, family-weighted",
                     "inside_rule": "distance <= k-NN radius", "git_commit": _git_commit(),
                     "python": platform.python_version(), "numpy": np.__version__},
        "datasets": datasets,
        "runtime_seconds": round(time.perf_counter() - started, 1),
    }
    for d in inputs:
        for crowd in report["datasets"][d.name]["crowds"].values():
            _assert_no_holdout(crowd["uncovered_ids"], d.splits, f"{d.name} output")
    (out_dir / "experiment0.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (out_dir / "experiment0.md").write_text(to_markdown(report), encoding="utf-8")
    return report


def load_inputs() -> list[DatasetInput]:
    """Real data, with holdout rows dropped immediately after loading."""
    from blindspot.data import gss, twin2k

    twin = twin2k.load()
    t_splits = load_splits("twin2k")
    twin_input = DatasetInput(
        "twin2k", "Twin-2K-500", twin["real"].subset(t_splits["pool"] + t_splits["test"]), t_splits,
        fixed_crowds={"gpt41mini_twin": twin["gpt41mini_twin"].subset(t_splits["test"]),
                      "retest": twin["retest"].subset(t_splits["test"])},
        map_crowds=("gpt41mini_twin", "gaussian_copula"))
    g = gss.load()
    g_splits = load_splits("gss")
    gss_input = DatasetInput("gss", "GSS 2024 (main cross-section, ballot 1)",
                             g.subset(g_splits["pool"] + g_splits["test"]), g_splits)
    return [twin_input, gss_input]


def main() -> None:
    report = run(load_inputs())
    print((RESULTS / "experiment0.md").read_text(encoding="utf-8"))
    print(f"total runtime {report['runtime_seconds']} s; per dataset (total / metrics without maps): "
          + ", ".join(f"{n} {d['runtime_seconds']} / {d['eval_seconds']} s" for n, d in report["datasets"].items()))


if __name__ == "__main__":
    main()
