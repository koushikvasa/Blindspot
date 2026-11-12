"""A fake Anthropic client for tests: counts every call, never touches the network."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np


def _system_text(params: dict) -> str:
    system = params.get("system", "")
    return system if isinstance(system, str) else "\n".join(b["text"] for b in system)


def _seed(params: dict) -> int:
    blob = _system_text(params) + json.dumps(params["messages"])
    return int(hashlib.sha256(blob.encode()).hexdigest()[:8], 16)


def persona_json(params: dict) -> str:
    """Valid persona answers for whatever questions the system prompt lists; varies with the prompt."""
    option_counts = [line.count(" = ") for line in _system_text(params).splitlines() if "Options:" in line]
    rng = np.random.default_rng(_seed(params))
    out = []
    for n, k in enumerate(option_counts, start=1):
        p = [100 // k] * (k - 1) + [100 - (100 // k) * (k - 1)]
        out.append({"q": n, "p": p, "a": int(rng.integers(0, k))})
    return json.dumps({"answers": out})


def generic_responder(params: dict, n: int) -> str:
    props = params["output_config"]["format"]["schema"]["properties"]
    if "answers" in props:
        return persona_json(params)
    if "description" in props:
        return json.dumps({"name": f"pattern {_seed(params) % 100}", "share_low_pct": 5, "share_high_pct": 20,
                           "description": "People whose judgments tend to move together in one direction, "
                                          "combining trust and caution in an unusual way."})
    if "segments" in props:
        return json.dumps({"segments": [
            {"conditions": [{"field": "age_band", "value": "65+"}], "reason": "older people show the pattern"},
            {"conditions": [{"field": "gender", "value": "Female"}, {"field": "region", "value": "South"}],
             "reason": "southern women show the pattern"}]})
    raise AssertionError(f"unexpected schema {list(props)}")


class FakeClient:
    """Stands in for anthropic.Anthropic."""

    def __init__(self, responder=generic_responder, input_tokens: int = 500):
        self.responder, self.calls, self.input_tokens = responder, [], input_tokens
        self.messages = SimpleNamespace(create=self._create, count_tokens=self._count, batches=self._batches())

    def with_options(self, **_):
        return self

    def _message(self, params):
        self.calls.append(params)
        text = self.responder(params, len(self.calls))
        usage = SimpleNamespace(input_tokens=100, output_tokens=50, cache_creation_input_tokens=0,
                                cache_read_input_tokens=0)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], stop_reason="end_turn",
                               model=params["model"], usage=usage)

    def _create(self, **params):
        return self._message(params)

    def _count(self, **_):
        return SimpleNamespace(input_tokens=self.input_tokens)

    def _batches(self):
        store = {}

        def create(requests):
            store["results"] = [SimpleNamespace(custom_id=r["custom_id"], result=SimpleNamespace(
                type="succeeded", message=self._message(dict(r["params"])))) for r in requests]
            return SimpleNamespace(id="batch_1", processing_status="ended")

        return SimpleNamespace(create=create, retrieve=lambda _id: None, results=lambda _id: store["results"])


def calls_by_kind(client: FakeClient) -> dict[str, int]:
    out = {"persona": 0, "describe": 0, "seek": 0}
    for params in client.calls:
        props = params["output_config"]["format"]["schema"]["properties"]
        out["persona" if "answers" in props else "describe" if "description" in props else "seek"] += 1
    return out
