"""Cached, cost-guarded Claude calls for the persona generator (H3).

- Every response is cached in results/cache/ (CLAUDE.md rule 7), keyed by a hash of
  (model, system prompt, user prompt, item codes, recipe version, replicate, attempt).
- Before any paid call, `estimate_cost` sizes the run and `check_cost` stops it above the limit.
- Calls go through the Message Batches API (50% price); `run_concurrent` is the fallback.
"""
import hashlib
import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "results/cache"
DEFAULT_COST_LIMIT_USD = 5.0
BATCH_POLL_SECONDS = 30
BATCH_TIMEOUT_SECONDS = 6 * 3600
CONCURRENCY = 8
BATCH_DISCOUNT = 0.5

# USD per million tokens (standard API). Unknown models are refused so the guard never guesses.
PRICING = {
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_write": 2.50, "cache_read": 0.10},
    "claude-opus-5-5": {"input": 4.00, "output": 20.00, "cache_write": 5.00, "cache_read": 0.20},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "cache_write": 0.125, "cache_read": 0.01},
}


class CostLimitExceeded(RuntimeError):
    """The estimated cost of a run is above the allowed limit; nothing was called."""


@dataclass(frozen=True)
class CallSpec:
    key: str                 # cache key (also the batch custom_id)
    params: dict             # Messages API request params


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float = 0.0
    batch: bool = True
    errors: list[str] = field(default_factory=list)

    def add(self, usage: dict, model: str, batch: bool) -> None:
        self.calls += 1
        self.input_tokens += usage.get("input_tokens", 0)
        self.output_tokens += usage.get("output_tokens", 0)
        self.cache_write_tokens += usage.get("cache_creation_input_tokens", 0) or 0
        self.cache_read_tokens += usage.get("cache_read_input_tokens", 0) or 0
        self.cost_usd += token_cost(usage, model, batch)

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def cache_key(**parts) -> str:
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:40]


class ResponseCache:
    def __init__(self, directory: Path = CACHE_DIR):
        self.dir = directory

    def path(self, key: str) -> Path:
        return self.dir / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict | None:
        p = self.path(key)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def put(self, key: str, record: dict) -> None:
        p = self.path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")


def _rates(model: str) -> dict:
    if model not in PRICING:
        raise ValueError(f"no pricing for model {model!r}; add it to PRICING before running")
    return PRICING[model]


def token_cost(usage: dict, model: str, batch: bool) -> float:
    r = _rates(model)
    cost = (usage.get("input_tokens", 0) * r["input"] + usage.get("output_tokens", 0) * r["output"]
            + (usage.get("cache_creation_input_tokens", 0) or 0) * r["cache_write"]
            + (usage.get("cache_read_input_tokens", 0) or 0) * r["cache_read"]) / 1e6
    return cost * (BATCH_DISCOUNT if batch else 1.0)


def estimate_cost(n_calls: int, input_tokens_per_call: int, output_tokens_per_call: int, model: str,
                  batch: bool = True) -> dict:
    """Conservative estimate: every input token billed at the uncached input rate."""
    usage = {"input_tokens": n_calls * input_tokens_per_call, "output_tokens": n_calls * output_tokens_per_call}
    return {"calls": n_calls, "input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"],
            "model": model, "batch": batch, "cost_usd": round(token_cost(usage, model, batch), 4)}


def check_cost(estimate: dict, limit_usd: float = DEFAULT_COST_LIMIT_USD) -> None:
    print(f"[cost guard] {estimate['calls']} calls, ~{estimate['input_tokens']:,} input + "
          f"~{estimate['output_tokens']:,} output tokens on {estimate['model']} "
          f"({'batch' if estimate['batch'] else 'standard'} pricing): ~${estimate['cost_usd']:.2f} "
          f"(limit ${limit_usd:.2f})")
    if estimate["cost_usd"] > limit_usd:
        raise CostLimitExceeded(f"estimated ${estimate['cost_usd']:.2f} exceeds the ${limit_usd:.2f} limit; "
                                "nothing was called. Re-run with a higher limit only after approval.")


def message_record(message) -> dict:
    """Serializable subset of a Messages API response."""
    text = "".join(b.text for b in message.content if getattr(b, "type", "") == "text")
    u = message.usage
    return {"text": text, "stop_reason": message.stop_reason, "model": message.model,
            "usage": {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                      "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
                      "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0}}


def run_batch(client, specs: list[CallSpec], log: Callable[[str], None] = print) -> dict[str, dict]:
    """Submit specs as one Message Batch, wait, and return {key: record or {'error': ...}}."""
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    batch = client.messages.batches.create(requests=[
        Request(custom_id=s.key, params=MessageCreateParamsNonStreaming(**s.params)) for s in specs])
    log(f"[batch] submitted {batch.id} with {len(specs)} requests")
    started = time.monotonic()
    while batch.processing_status != "ended":
        if time.monotonic() - started > BATCH_TIMEOUT_SECONDS:
            raise TimeoutError(f"batch {batch.id} did not finish in {BATCH_TIMEOUT_SECONDS} s")
        time.sleep(BATCH_POLL_SECONDS)
        batch = client.messages.batches.retrieve(batch.id)
        c = batch.request_counts
        log(f"[batch] {batch.processing_status}: {c.succeeded} ok, {c.errored} errored, {c.processing} processing")
    out = {}
    for result in client.messages.batches.results(batch.id):
        if result.result.type == "succeeded":
            out[result.custom_id] = message_record(result.result.message)
        else:
            detail = getattr(getattr(result.result, "error", None), "type", "") or result.result.type
            out[result.custom_id] = {"error": f"{result.result.type}:{detail}"}
    return out


def run_concurrent(client, specs: list[CallSpec], workers: int = CONCURRENCY) -> dict[str, dict]:
    """Fallback path: bounded concurrency; the SDK retries 429/5xx with backoff."""
    import anthropic

    def one(spec: CallSpec) -> tuple[str, dict]:
        try:
            return spec.key, message_record(client.with_options(max_retries=5).messages.create(**spec.params))
        except anthropic.APIStatusError as e:
            return spec.key, {"error": f"status_{e.status_code}: {str(e.message)[:300]}"}
        except anthropic.APIConnectionError:
            return spec.key, {"error": "connection"}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(one, specs))


def call_cached(client, specs: list[CallSpec], cache: ResponseCache, model: str, usage: Usage,
                use_batch: bool = True, log: Callable[[str], None] = print) -> dict[str, dict]:
    """Return records for all specs, calling the API only for cache misses."""
    records = {s.key: cache.get(s.key) for s in specs}
    missing = [s for s in specs if records[s.key] is None]
    log(f"[cache] {len(specs) - len(missing)} hits, {len(missing)} to call")
    if missing:
        fresh = run_batch(client, missing, log) if use_batch else run_concurrent(client, missing)
        for key, rec in fresh.items():
            if "error" in rec:
                usage.errors.append(f"{key}:{rec['error']}")
                continue
            usage.add(rec["usage"], model, use_batch)
            cache.put(key, rec)
            records[key] = rec
    return records
