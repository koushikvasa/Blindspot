"""Prepared data, Gower distance and the small-sample coverage estimators (numpy only).

Coverage follows the research metric (Naeem et al., 2020): a real person is COVERED when some AI person is at least as
close to them as their k-th most similar OTHER real person (so each real person is left out of their own radius).
A small real sample makes those radii too large and the AI panel look better than it is, so the estimator can rescale the
radius to what a larger real sample would give (`corrected`); `calibration.json` (written by the validation run) holds the
estimator that tracked the truth best and the error table shown in the report.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .match import is_metadata, match_columns
from .parse import AuditError, Table, is_missing
from .schema import ColumnSpec, analyse_column

MIN_REAL = 10
MAX_REAL = 500
CALIBRATION = Path(__file__).with_name("calibration.json")
DEFAULT_CALIBRATION = {"estimator": {"k": 3, "corrected": True}, "n_ref": 334, "k_ref": 5, "error_table": {},
                       "min_reliable_n": 50}


def load_calibration() -> dict:
    try:
        return {**DEFAULT_CALIBRATION, **json.loads(CALIBRATION.read_text(encoding="utf-8"))}
    except (OSError, ValueError):
        return dict(DEFAULT_CALIBRATION)


@dataclass
class Prepared:
    columns: list[ColumnSpec]
    entries: list[dict]                       # every matched column, with the reason it is off by default
    unmatched_panel: list[str]
    unmatched_real: list[str]
    X_real: np.ndarray
    X_syn: np.ndarray
    cat: np.ndarray
    raw_real: list[list[str]]
    raw_syn: list[list[str]]
    n_real_file: int
    n_syn: int
    notes: list[str] = field(default_factory=list)


def inspect_columns(panel: Table, real: Table) -> tuple[list[dict], list[str], list[str]]:
    matches, un_panel, un_real = match_columns(panel.headers, real.headers)
    pidx = {h: i for i, h in enumerate(panel.headers)}
    ridx = {h: i for i, h in enumerate(real.headers)}
    entries = []
    for m in matches:
        spec, why = analyse_column(m.real, real.column(ridx[m.real]), panel.column(pidx[m.panel]))
        reason = why or ("looks like metadata (id, time, status, ...)" if is_metadata(m.real) else None)
        entries.append({"name": m.real, "panel_header": m.panel, "method": m.method, "score": m.score, "spec": spec,
                        "reason": reason, "p_idx": pidx[m.panel], "r_idx": ridx[m.real]})
    return entries, un_panel, un_real


def _encode_column(spec: ColumnSpec, values: list[str]) -> np.ndarray:
    cache: dict[str, float] = {}
    out = np.empty(len(values))
    for i, v in enumerate(values):
        if v not in cache:
            cache[v] = spec.encode(v)
        out[i] = cache[v]
    return out


def prepare(panel: Table, real: Table, use: set[str] | None = None, seed: int = 42) -> Prepared:
    entries, un_panel, un_real = inspect_columns(panel, real)
    if not entries:
        raise AuditError("No question columns match between the two files. The column headers must be the same "
                         "(small differences such as capitals or underscores are fine).")
    usable = [e for e in entries if e["spec"] is not None]
    chosen = [e for e in usable if (e["reason"] is None if use is None else e["name"] in use)]
    if not chosen:
        raise AuditError("None of the matched columns can be compared (they look like ids, free text or constants). "
                         "Tick at least one question column.")
    notes: list[str] = []
    real_rows = list(real.rows)
    if len(real_rows) < MIN_REAL:
        raise AuditError(f"Your real sample has only {len(real_rows)} rows; at least {MIN_REAL} are needed "
                         "(25 or more is recommended, see the expected error in the report).")
    n_file = len(real_rows)
    if n_file > MAX_REAL:
        keep = np.sort(np.random.default_rng(seed).choice(n_file, MAX_REAL, replace=False))
        real_rows = [real_rows[i] for i in keep]
        notes.append(f"Your real sample has {n_file:,} rows; a random {MAX_REAL} were used.")
    specs = [e["spec"] for e in chosen]
    X_real = np.column_stack([_encode_column(e["spec"], [r[e["r_idx"]] for r in real_rows]) for e in chosen])
    X_syn = np.column_stack([_encode_column(e["spec"], panel.column(e["p_idx"])) for e in chosen])
    unknown = float(np.isnan(X_syn).mean())
    if unknown > 0.5:
        raise AuditError("Most answers in the AI panel could not be read for the matched columns.")
    return Prepared(specs, entries, un_panel, un_real, X_real, X_syn, np.array([s.kind == "categorical" for s in specs]),
                    [[r[e["r_idx"]] for e in chosen] for r in real_rows],
                    [[r[e["p_idx"]] for e in chosen] for r in panel.rows], n_file, panel.n_rows, notes)


def gower(A: np.ndarray, B: np.ndarray, cat: np.ndarray) -> np.ndarray:
    """Distance matrix len(A) x len(B): mean over the questions both people answered (equal weight per question);
    categorical = 0/1 mismatch, ordinal/numeric = absolute difference of the [0, 1] scale. No overlap -> 1."""
    num = np.zeros((len(A), len(B)), dtype=np.float32)
    den = np.zeros_like(num)
    for j in range(A.shape[1]):
        a, b = A[:, j][:, None], B[:, j][None, :]
        both = ~np.isnan(a) & ~np.isnan(b)
        diff = (a != b).astype(np.float32) if cat[j] else np.abs(a - b).astype(np.float32)
        num += np.where(both, diff, 0.0)
        den += both
    return np.where(den > 0, num / np.maximum(den, 1), 1.0).astype(np.float32)


@dataclass(frozen=True)
class Estimator:
    k: int = 3
    corrected: bool = False
    power: float = 1.0          # how much of the sample-size rescaling to apply (1 = the full dimension-based correction)


def intrinsic_dimension(d_rr: np.ndarray) -> float:
    """From how the k-NN radius grows with k among the real people (radius ~ k^(1/d))."""
    n = len(d_rr)
    ks = [k for k in range(1, 6) if k < n]
    if len(ks) < 2:
        return 8.0
    r = np.sort(d_rr, axis=1)
    med = np.array([np.median(r[:, k - 1]) for k in ks])
    ok = med > 0
    if ok.sum() < 2:
        return 8.0
    slope = np.polyfit(np.log(np.array(ks)[ok]), np.log(med[ok]), 1)[0]
    return float(np.clip(1.0 / slope, 2.0, 40.0)) if slope > 0 else 8.0


def radii(d_rr: np.ndarray, est: Estimator, n_ref: int = 334, k_ref: int = 5) -> np.ndarray:
    """k-th nearest OTHER real person, optionally rescaled to what a reference-size (n_ref) real sample would give."""
    n = len(d_rr)
    k = max(1, min(est.k, n - 1))
    r = np.partition(d_rr, k - 1, axis=1)[:, k - 1].astype(np.float64)
    if est.corrected:
        scale = ((k_ref / (n_ref - 1)) / (k / (n - 1))) ** (1.0 / intrinsic_dimension(d_rr))
        r = r * float(np.clip(scale, 0.25, 2.0)) ** est.power
    return r


def with_self_excluded(d_rr: np.ndarray) -> np.ndarray:
    out = d_rr.astype(np.float64).copy()
    np.fill_diagonal(out, np.inf)
    return out


def coverage_precision(d_rs: np.ndarray, radius: np.ndarray) -> tuple[float, float, np.ndarray]:
    covered = d_rs.min(axis=1) <= radius
    precision = float((d_rs <= radius[:, None]).any(axis=0).mean())
    return float(covered.mean()), precision, covered


def estimate(prep: Prepared, est: Estimator, cal: dict | None = None) -> dict:
    cal = cal or load_calibration()
    d_rs = gower(prep.X_real, prep.X_syn, prep.cat)
    d_rr = with_self_excluded(gower(prep.X_real, prep.X_real, prep.cat))
    radius = radii(d_rr, est, cal["n_ref"], cal["k_ref"])
    coverage, _, covered = coverage_precision(d_rs, radius)
    p_est = Estimator(**cal.get("precision_estimator", cal["estimator"]))
    _, precision, _ = coverage_precision(d_rs, radii(d_rr, p_est, cal["n_ref"], cal["k_ref"]))
    return {"coverage": coverage, "precision": precision, "covered": covered, "radius": radius, "d_rs": d_rs, "d_rr": d_rr}


def expected_error(n: int, cal: dict | None = None) -> dict:
    """Typical error of the coverage estimate for a real sample of n people (from the validation run)."""
    cal = cal or load_calibration()
    table = {int(k): v for k, v in cal.get("error_table", {}).items()}
    if not table:
        return {"n": n, "rmse": None, "bias": None, "sd": None, "reliable": n >= cal["min_reliable_n"],
                "min_reliable_n": cal["min_reliable_n"], "table_n": None}
    nearest = min(table, key=lambda m: abs(m - n))
    row = table[nearest]
    return {"n": n, "rmse": row["rmse"], "bias": row["bias"], "sd": row["sd"], "table_n": nearest,
            "reliable": n >= cal["min_reliable_n"], "min_reliable_n": cal["min_reliable_n"]}
