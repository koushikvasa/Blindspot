"""2-D map coordinates for the UI (H2): UMAP on the precomputed Gower distance of real + crowd."""
import json
from pathlib import Path

import numpy as np

from blindspot.data.schema import Respondents

from .gower import encode, gower

ROOT = Path(__file__).resolve().parents[2]
MAP_DIR = ROOT / "results/maps"
NO_OVERLAP_DISTANCE = 1.0  # two people with no shared answered item are treated as maximally far


def _umap_xy(dist: np.ndarray, seed: int) -> np.ndarray:
    import umap  # slow import (numba); only needed here

    n_neighbors = max(2, min(15, len(dist) - 1))
    # compatibility_layout=True pins the classic UMAP layout pipeline so maps stay stable across versions.
    reducer = umap.UMAP(metric="precomputed", n_neighbors=n_neighbors, min_dist=0.1, random_state=seed,
                        compatibility_layout=True)
    return reducer.fit_transform(dist)


def map_coords(real: Respondents, synthetic_by_name: dict[str, Respondents], seed: int = 42,
               covered_by_name: dict[str, dict[str, np.ndarray]] | None = None,
               dataset: str = "dataset", out_dir: Path = MAP_DIR) -> dict[str, Path]:
    """Write results/maps/{dataset}_{crowd}.json: [{id, x, y, kind, crowd, covered}, ...].

    `covered` for a real point = covered by the crowd; for a synthetic point = inside at least one
    real person's k-NN ball (the precision criterion). Taken from coverage.evaluate output.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    real_x = encode(real)
    paths = {}
    for name, crowd in synthetic_by_name.items():
        combined = np.vstack([real_x, encode(crowd)])
        dist = np.nan_to_num(gower(combined, combined, real.items), nan=NO_OVERLAP_DISTANCE)
        np.fill_diagonal(dist, 0.0)
        xy = _umap_xy(dist, seed)
        flags = (covered_by_name or {}).get(name, {})
        real_cov = flags.get("real_covered", np.full(len(real.answers), None))
        syn_cov = flags.get("synthetic_inside", np.full(len(crowd.answers), None))
        rows = [{"id": str(i), "x": float(x), "y": float(y), "kind": "real", "crowd": name,
                 "covered": None if c is None else bool(c)}
                for i, (x, y), c in zip(real.answers.index, xy[:len(real_x)], real_cov)]
        rows += [{"id": str(i), "x": float(x), "y": float(y), "kind": "synthetic", "crowd": name,
                  "covered": None if c is None else bool(c)}
                 for i, (x, y), c in zip(crowd.answers.index, xy[len(real_x):], syn_cov)]
        path = out_dir / f"{dataset}_{name}.json"
        path.write_text(json.dumps(rows), encoding="utf-8")
        paths[name] = path
    return paths
