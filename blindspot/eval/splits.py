"""Seeded pool/test/holdout split of real respondents (H1).

pool    : the agent may request examples from these people
test    : round-by-round scores
holdout : headline number, read only by eval/final.py

Splits are made once per dataset and saved to data/processed/splits_{dataset}.json. For Twin the
one split applies to real, gpt41mini_twin and retest (they share ids).
"""
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / "data/processed"
SPLIT_NAMES = ("pool", "test", "holdout")


class LeakError(AssertionError):
    """Raised when ids outside the allowed splits are about to be used."""


def make_splits(ids: Iterable[str], seed: int = 42,
                fractions: tuple[float, float, float] = (0.5, 0.3, 0.2)) -> dict[str, list[str]]:
    """Deterministic, disjoint split. Ids are sorted first, so input order does not matter.

    pool and test sizes are floor(n * fraction); holdout takes the remainder.
    """
    if len(fractions) != 3 or any(f <= 0 for f in fractions) or not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"fractions must be three positive numbers summing to 1, got {fractions}")
    ordered = sorted({str(i) for i in ids})
    if len(ordered) < 3:
        raise ValueError("need at least 3 unique ids")
    perm = np.random.default_rng(seed).permutation(len(ordered))
    shuffled = [ordered[i] for i in perm]
    n_pool = int(len(shuffled) * fractions[0])
    n_test = int(len(shuffled) * fractions[1])
    return {
        "pool": sorted(shuffled[:n_pool]),
        "test": sorted(shuffled[n_pool:n_pool + n_test]),
        "holdout": sorted(shuffled[n_pool + n_test:]),
    }


def ids_fingerprint(ids: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode()).hexdigest()[:16]


def split_path(dataset: str, split_dir: Path = SPLIT_DIR) -> Path:
    return split_dir / f"splits_{dataset}.json"


def save_splits(dataset: str, splits: dict[str, list[str]], seed: int,
                fractions: tuple[float, float, float], split_dir: Path = SPLIT_DIR) -> Path:
    all_ids = [i for name in SPLIT_NAMES for i in splits[name]]
    payload = {
        "dataset": dataset, "seed": seed, "fractions": list(fractions),
        "n": len(all_ids), "ids_fingerprint": ids_fingerprint(all_ids),
        **{name: splits[name] for name in SPLIT_NAMES},
    }
    path = split_path(dataset, split_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return path


def load_splits(dataset: str, split_dir: Path = SPLIT_DIR) -> dict[str, list[str]]:
    payload = json.loads(split_path(dataset, split_dir).read_text(encoding="utf-8"))
    return {name: payload[name] for name in SPLIT_NAMES}


def get_or_make_splits(dataset: str, ids: Iterable[str], seed: int = 42,
                       fractions: tuple[float, float, float] = (0.5, 0.3, 0.2),
                       split_dir: Path = SPLIT_DIR) -> dict[str, list[str]]:
    """Load the saved split; create it once if missing. Refuses if the saved split covers other ids."""
    ids = list(ids)
    if split_path(dataset, split_dir).exists():
        splits = load_splits(dataset, split_dir)
        saved = [i for name in SPLIT_NAMES for i in splits[name]]
        if ids_fingerprint(saved) != ids_fingerprint(ids):
            raise ValueError(f"saved split for {dataset!r} covers different ids; delete it deliberately to redo")
        return splits
    splits = make_splits(ids, seed, fractions)
    save_splits(dataset, splits, seed, fractions, split_dir)
    return splits


def assert_no_leak(ids_used: Iterable[str], splits: dict[str, list[str]],
                   allowed: tuple[str, ...] = ("pool",)) -> None:
    """Raise LeakError if any id in `ids_used` belongs to a split not in `allowed` (or to no split)."""
    unknown = [a for a in allowed if a not in SPLIT_NAMES]
    if unknown:
        raise ValueError(f"unknown split names {unknown}")
    permitted = {i for name in allowed for i in splits[name]}
    leaked = sorted({str(i) for i in ids_used} - permitted)
    if leaked:
        where = {name: len(set(leaked) & set(splits[name])) for name in SPLIT_NAMES}
        raise LeakError(f"{len(leaked)} ids outside {allowed}: by split {where}, e.g. {leaked[:3]}")
