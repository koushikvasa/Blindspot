"""Write the sample files for the audit page: Flashpoint synthetic panel 1 and 100 random GSS POOL people (20 live items).

`python -m blindspot.eval.audit_samples`
Both files hold answer LABELS as text and the full question wording as column headers, exactly like a real export, and no ids. The real sample comes from the POOL split
only (never test or holdout). Output: blindspot/audit/sample/{ai_panel_flashpoint.csv, real_sample_gss_100.csv}.
"""
import csv
import json
from pathlib import Path

import numpy as np

from blindspot.data import gss
from blindspot.generators.flashpoint_panel import load as load_flashpoint

from .experiment_fp import FP_DIR, RUNS
from .splits import load_splits

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "blindspot" / "audit" / "sample"
N_REAL = 100
SEED = 42


def main() -> None:
    g = gss.load()
    live = [i for i in g.items if i.live_ok]
    pool = g.subset(load_splits("gss")["pool"])
    ids = np.array(pool.answers.index)
    pick = np.sort(np.random.default_rng(SEED).choice(len(ids), N_REAL, replace=False))
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "real_sample_gss_100.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([i.wording for i in live])
        for pos in pick:
            row = pool.answers.iloc[pos]
            w.writerow(["" if np.isnan(row[i.code]) else i.options[int(row[i.code])] for i in live])
    meta = json.loads((FP_DIR / "runs_meta.json").read_text(encoding="utf-8"))
    fp, _ = load_flashpoint(FP_DIR / meta["runs"][RUNS[0]]["rows_file"], live, RUNS[0])
    with (OUT / "ai_panel_flashpoint.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([i.wording for i in live])
        for _, row in fp.answers.iterrows():
            w.writerow([i.options[int(row[i.code])] for i in live])
    print(f"wrote {len(fp.answers)} AI rows and {N_REAL} real rows to {OUT}")


if __name__ == "__main__":
    main()
