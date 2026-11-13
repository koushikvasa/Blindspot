"""The redesigned loop (H4b): detect -> propose -> verify -> escalate -> assemble -> measure.

variant "verify"   : the full design (a recipe is kept only if it makes synthetic people closer to the cluster).
variant "ablation" : step 3 skipped, every valid recipe accepted (so nothing is ever escalated and asking
                     stays random: the ablation removes verification AND the targeting it triggers).
The agent asks pool people only; test is used only to score; holdout is never read.
Round 1 always asks at random. Model-written demographic reasons are not used anywhere in this loop.
"""
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from blindspot.agent.asking import draw_random, draw_targeted
from blindspot.agent.assemble import assemble
from blindspot.agent.book import ClusterBook
from blindspot.agent.budget import Budget
from blindspot.agent.describe import ClaudeCaller
from blindspot.agent.detect import detection_counts
from blindspot.agent.escalate import choose_segment
from blindspot.agent.gaps import DEFAULT_K, cluster_uncovered
from blindspot.agent.generate import PersonaGenerator
from blindspot.agent.guards import assert_few_near_duplicates, near_duplicates
from blindspot.agent.library import DEFAULT_MERGE_DISTANCE, centroid
from blindspot.agent.loop import LoopResult, _public
from blindspot.agent.propose import propose_variants
from blindspot.agent.verify import N_VERIFY, verify_variants
from blindspot.coverage import evaluate
from blindspot.data.schema import Respondents
from blindspot.eval.splits import assert_no_leak
from blindspot.generators.claude_client import ResponseCache, Usage
from blindspot.space import encode

VARIANTS_B = ("verify", "ablation")
UNVERIFIED_NOTE = ("no model-written demographic reasons are generated in this loop; any such text from "
                   "earlier loops is unverified and must never be shown in the UI")


@dataclass(frozen=True)
class LoopConfigB:
    dataset: str
    seed: int
    variant: str = "verify"
    rounds: int = 4
    per_round: int = 25
    k: int = DEFAULT_K
    n_verify: int = N_VERIFY
    max_attempts: int = 2
    merge_distance: float = DEFAULT_MERGE_DISTANCE
    n_boot: int = 50
    out_dir: Path | None = None

    def __post_init__(self):
        if self.variant not in VARIANTS_B:
            raise ValueError(f"variant must be one of {VARIANTS_B}")


def _handle_cluster(cfg, r: int, ci: int, ids: list[str], book: ClusterBook, pool: Respondents, items, encoded,
                    caller, gen, splits, n_asked: int) -> tuple[dict, str | None]:
    """Merge a cluster into an accepted entry, or run propose/verify (up to max_attempts). Returns (log, id of
    the entry if it ended as 'cannot imagine')."""
    match, dist = book.find_match(centroid(np.vstack([encoded[i] for i in ids])))
    log = {"size": len(ids), "ids": ids}
    if match is not None and not match.cannot_imagine:
        book.merge_into(match, ids)
        log.update(action="merged", entry=match.id, centroid_distance=round(dist, 4))
        return log, None
    if match is not None:                       # a known "cannot imagine" cluster with new data: try again
        book.merge_into(match, ids)
        entry = match
    else:
        entry = book.new_entry(ids, r)
    members = pool.subset(sorted(entry.members))
    attempts, rejected, accepted = [], [], []
    for attempt in range(1, cfg.max_attempts + 1):
        variants, dropped = propose_variants(caller, items, members, n_asked, splits, attempt, rejected)
        rng = np.random.default_rng([cfg.seed, 31, r, ci, attempt])
        rows = verify_variants(gen, variants, members, items, rng, cfg.n_verify,
                               accept_all=cfg.variant == "ablation")
        attempts.append({"attempt": attempt, "n_proposed": len(variants), "dropped_by_guard": dropped,
                         "variants": [{**row, "text": v.text} for v, row in zip(variants, rows)]})
        entry.attempts += 1
        accepted = [(v, row) for v, row in zip(variants, rows) if row["accepted"]]
        if accepted:
            break
        rejected = [v.name for v in variants]
    entry.variants = [v for v, _ in accepted]
    entry.variant_stats = [row for _, row in accepted]
    log.update(entry=entry.id, attempts=attempts, n_accepted=len(accepted),
               action="accepted" if accepted else "cannot_imagine")
    return log, None if accepted else entry.id


def run_loop_b(cfg: LoopConfigB, real: Respondents, splits: dict[str, list[str]], baseline: Respondents,
               rr: np.ndarray, client, cache: ResponseCache, model: str, budget: Budget,
               log: Callable[[str], None] = print) -> LoopResult:
    """`real` must contain pool + test people only (holdout already dropped)."""
    assert_no_leak(real.answers.index, splits, allowed=("pool", "test"))
    items = real.items
    pool, target = real.subset(splits["pool"]), real.subset(splits["test"])
    encoded = dict(zip(pool.answers.index, encode(pool)))
    tag = f"{cfg.dataset}/{cfg.variant}/seed{cfg.seed}"
    usage = Usage(batch=False)
    budget.register(usage)
    caller = ClaudeCaller(client, cache, model, usage, log)
    gen = PersonaGenerator(client, cache, model, usage, budget, cfg.dataset, items, cfg.seed, tag, log)
    book = ClusterBook(items, cfg.merge_distance)
    ask_rng = np.random.default_rng([cfg.seed, 21])
    result = LoopResult(cfg, usage=usage)
    asked: list[str] = []
    crowd, next_segment, escalation = baseline, None, {"triggered": False, "next_round": "random (round 1)"}

    for r in range(1, cfg.rounds + 1):
        started = time.perf_counter()
        used_escalation = escalation
        if next_segment is None:
            new_ids, asking_report = draw_random(list(pool.answers.index), set(asked), cfg.per_round, ask_rng), []
        else:
            new_ids, asking_report = draw_targeted(pool.demographics, set(asked), [next_segment], cfg.per_round,
                                                   ask_rng)
        assert_no_leak(new_ids, splits, allowed=("pool",))
        asked = sorted(asked + new_ids)
        asked_resp = pool.subset(asked)

        uncovered, detection = detection_counts(asked_resp, crowd, items, cfg.k, cfg.seed * 100 + r)
        clusters = cluster_uncovered(asked_resp, uncovered, items)
        cluster_logs, cannot = [], []
        for ci, ids in enumerate(clusters):
            entry_log, stuck = _handle_cluster(cfg, r, ci, ids, book, pool, items, encoded, caller, gen, splits,
                                               len(asked))
            cluster_logs.append(entry_log)
            cannot += [stuck] if stuck else []
        book.refresh(encoded)

        next_segment, escalation = (choose_segment(asked_resp.demographics, set(uncovered), cannot)
                                    if r < cfg.rounds else (None, {"triggered": False, "next_round": "n/a (last)"}))
        crowd, assembly = assemble(gen, baseline, book, pool, len(asked),
                                   np.random.default_rng([cfg.seed, 41, r]))

        metrics = evaluate(target, crowd, items, k=cfg.k, n_boot=cfg.n_boot, seed=cfg.seed,
                           k_values=(3, 5, 10), real_dist=rr)
        dup = near_duplicates(crowd, asked_resp)
        assert_few_near_duplicates(dup)
        snapshot = book.snapshot(len(asked))
        if cfg.out_dir is not None:
            cfg.out_dir.mkdir(parents=True, exist_ok=True)
            (cfg.out_dir / f"clusters_v{r}.json").write_text(json.dumps(snapshot, indent=1), encoding="utf-8")
        result.crowds.append(crowd)
        result.asked_by_round.append(list(asked))
        result.private_eval.append({"real_covered": metrics["_real_covered"],
                                    "synthetic_inside": metrics["_synthetic_inside"]})
        n_variants = sum(a["n_proposed"] for c in cluster_logs for a in c.get("attempts", []))
        n_accepted = sum(c.get("n_accepted", 0) for c in cluster_logs)
        result.rounds.append({
            "round": r, "budget": len(asked), "asked_new": new_ids, "asking_mode":
                "random" if not asking_report else "targeted", "asking_report": asking_report,
            "escalation_used_for_this_round": used_escalation,
            "detection": detection, "uncovered_asked": uncovered, "clusters": cluster_logs,
            "n_variants_proposed": n_variants, "n_variants_accepted": n_accepted,
            "n_clusters_cannot_imagine": len(cannot), "escalation": escalation, "book": snapshot,
            "assembly": assembly, "metrics": _public(metrics),
            "near_duplicates": {"share": dup.share, "n": dup.n_duplicates, "threshold": dup.threshold},
            "model_written_demographic_reasons": UNVERIFIED_NOTE,
            "cost_usd_cumulative": round(usage.cost_usd, 4), "seconds": round(time.perf_counter() - started, 1)})
        log(f"[{tag}] round {r}: asked {len(asked)}, uncovered new/old {detection['n_uncovered_new_rule']}/"
            f"{detection['n_uncovered_old_rule']}, clusters {len(clusters)}, accepted {n_accepted}/{n_variants}, "
            f"replaced {assembly['n_replaced']}, coverage {metrics['coverage']:.3f}, precision "
            f"{metrics['precision']:.3f}, dup {dup.share:.1%}, ${usage.cost_usd:.3f}")
    return result
