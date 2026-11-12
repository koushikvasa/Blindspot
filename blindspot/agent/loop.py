"""The self-improving loop (H4): ask -> find gaps -> describe -> seek -> learn -> regenerate -> measure.

The agent starts with its synthetic crowd (claude_mode, budget 0) and a budget of real people it may
ask, drawn from the POOL split. It never sees the test split; every round is scored on test.
Deviations from the handover order, all deliberate:
- Learn runs before Seek (seeking needs archetype membership to look up members' demographics).
- A new cluster whose answer-pattern centroid is near an existing archetype is merged BEFORE any
  Claude call (cheaper than describing and then merging by text).
"""
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from blindspot.agent.asking import Segment, draw_random, draw_targeted
from blindspot.agent.budget import Budget
from blindspot.agent.describe import ClaudeCaller, demographic_counts, describe_cluster, seek_segments
from blindspot.agent.gaps import DEFAULT_K, cluster_uncovered, uncovered_asked
from blindspot.agent.guards import assert_few_near_duplicates, near_duplicates
from blindspot.agent.library import DEFAULT_MERGE_DISTANCE, RecipeLibrary, centroid
from blindspot.coverage import evaluate
from blindspot.data.schema import Respondents
from blindspot.eval.splits import assert_no_leak
from blindspot.generators import claude_personas as cp
from blindspot.generators.claude_client import ResponseCache, Usage
from blindspot.space import encode

VARIANTS = ("targeted", "random")


@dataclass(frozen=True)
class LoopConfig:
    dataset: str
    seed: int
    variant: str = "targeted"
    rounds: int = 4
    per_round: int = 25
    k: int = DEFAULT_K
    max_archetype_share: float = 1.0       # cap on the share of synthetic people given an archetype
    merge_distance: float = DEFAULT_MERGE_DISTANCE
    n_boot: int = 50
    batch: bool = True
    out_dir: Path | None = None            # where recipes_v{r}.json are written

    def __post_init__(self):
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")


@dataclass
class LoopResult:
    config: LoopConfig
    rounds: list[dict] = field(default_factory=list)
    crowds: list[Respondents] = field(default_factory=list)          # crowd after each round
    asked_by_round: list[list[str]] = field(default_factory=list)    # cumulative asked ids per round
    private_eval: list[dict] = field(default_factory=list)           # covered flags (for maps)
    usage: Usage = field(default_factory=Usage)


def _public(metrics: dict) -> dict:
    out = {k: v for k, v in metrics.items() if not k.startswith("_") and k != "uncovered_ids"}
    out["n_uncovered"] = len(metrics["uncovered_ids"])
    return out


def run_loop(cfg: LoopConfig, real: Respondents, splits: dict[str, list[str]], baseline: Respondents,
             rr: np.ndarray, client, cache: ResponseCache, model: str, budget: Budget,
             log: Callable[[str], None] = print) -> LoopResult:
    """`real` must contain pool + test people only (holdout already dropped)."""
    assert_no_leak(real.answers.index, splits, allowed=("pool", "test"))
    items = real.items
    pool, target = real.subset(splits["pool"]), real.subset(splits["test"])
    encoded = dict(zip(pool.answers.index, encode(pool)))
    rng = np.random.default_rng([cfg.seed, 11])                       # asking
    uniforms = np.random.default_rng([cfg.seed, 7]).random(len(target.answers))   # fixed archetype draws
    usage = Usage(batch=cfg.batch)
    budget.register(usage)
    caller = ClaudeCaller(client, cache, model, usage, log)
    lib = RecipeLibrary(items, cfg.merge_distance)
    result = LoopResult(cfg, usage=usage)
    asked: list[str] = []
    crowd, segments = baseline, []
    tag = f"{cfg.dataset}/{cfg.variant}/seed{cfg.seed}"

    for r in range(1, cfg.rounds + 1):
        started = time.perf_counter()
        # 1. ask
        used_segments = segments
        if r == 1 or cfg.variant == "random":
            new_ids, segment_report = draw_random(list(pool.answers.index), set(asked), cfg.per_round, rng), []
        else:
            new_ids, segment_report = draw_targeted(pool.demographics, set(asked), segments, cfg.per_round, rng)
        assert_no_leak(new_ids, splits, allowed=("pool",))
        asked = sorted(asked + new_ids)
        asked_resp = pool.subset(asked)

        # 2. find gaps  3. describe  5. learn
        uncovered = uncovered_asked(asked_resp, crowd, items, cfg.k, seed=cfg.seed * 100 + r)
        clusters = cluster_uncovered(asked_resp, uncovered, items)
        cluster_logs = []
        for ids in clusters:
            match, dist = lib.find_match(centroid(np.vstack([encoded[i] for i in ids])))
            entry = {"size": len(ids), "ids": ids}
            if match is not None:
                lib.merge_into(match, ids)
                entry.update(action="merged", archetype=match.id, centroid_distance=round(dist, 4))
            else:
                desc = describe_cluster(caller, items, pool.subset(ids), len(asked), splits)
                if desc is None or desc.discarded:
                    entry.update(action="discarded", problems=desc.problems if desc else ["no response"])
                else:
                    a = lib.add(desc.name, desc.text, ids, r, desc.share_low, desc.share_high)
                    entry.update(action="described", archetype=a.id, name=a.name, attempts=desc.attempts)
            cluster_logs.append(entry)
        lib.refresh(encoded)
        if cfg.out_dir is not None:
            lib.save(cfg.out_dir / f"recipes_v{r}.json", len(asked), cfg.max_archetype_share)

        # 4. seek (targeted variant only; the last round has no next round)
        next_segments: list[Segment] = []
        if cfg.variant == "targeted" and r < cfg.rounds and lib.archetypes:
            summaries = [{"id": a.id, "name": a.name, "text": a.text, "n_members": len(a.members),
                          "demographics": demographic_counts(asked_resp.demographics.loc[sorted(a.members)],
                                                             cp.PERSONA_FIELDS)} for a in lib.archetypes]
            next_segments = seek_segments(caller, summaries, asked_resp.demographics, pool.demographics,
                                          asked, splits)
        segments = next_segments

        # 6. regenerate (mode answers), guarded per round
        recipes = lib.recipes_for(uniforms, len(asked), cfg.max_archetype_share)
        persona_plan = cp.plan(cfg.dataset, target.demographics, items, model, recipes=recipes)
        estimate = cp.estimate(persona_plan, client, cache, cfg.batch)
        budget.check(estimate["cost_usd"], f"{tag} round {r}")
        crowd = cp.execute(persona_plan, client, cache, usage, seed=cfg.seed, batch=cfg.batch).mode

        # 7. measure (+ copy guard)
        metrics = evaluate(target, crowd, items, k=cfg.k, n_boot=cfg.n_boot, seed=cfg.seed,
                           k_values=(3, 5, 10), real_dist=rr)
        dup = near_duplicates(crowd, asked_resp)
        assert_few_near_duplicates(dup)
        result.crowds.append(crowd)
        result.asked_by_round.append(list(asked))
        result.private_eval.append({"real_covered": metrics["_real_covered"],
                                    "synthetic_inside": metrics["_synthetic_inside"]})
        result.rounds.append({
            "round": r, "budget": len(asked), "asked_new": new_ids,
            "segments_requested": [s.to_dict() for s in used_segments], "asking_report": segment_report,
            "uncovered_asked": uncovered, "clusters": cluster_logs,
            "library": lib.snapshot(len(asked), cfg.max_archetype_share), "library_version": lib.version,
            "segments_proposed_next": [s.to_dict() for s in next_segments],
            "n_synthetic_with_archetype": int(sum(1 for x in recipes if x)),
            "metrics": _public(metrics),
            "near_duplicates": {"share": dup.share, "n": dup.n_duplicates, "threshold": dup.threshold},
            "estimate_usd": estimate["cost_usd"], "cost_usd_cumulative": round(usage.cost_usd, 4),
            "seconds": round(time.perf_counter() - started, 1)})
        log(f"[{tag}] round {r}: asked {len(asked)}, {len(uncovered)} uncovered, "
            f"{len(lib.archetypes)} archetypes, coverage {metrics['coverage']:.3f}, "
            f"precision {metrics['precision']:.3f}, dup {dup.share:.1%}, ${usage.cost_usd:.3f}")
    return result
