import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from blindspot.data.schema import Item, Respondents
from blindspot.eval.experiment0 import DatasetInput
from blindspot.eval.experiment0 import run as run_experiment0
from blindspot.eval.experiment1 import bunching_share, guard
from blindspot.eval.experiment1 import run as run_experiment1
from blindspot.eval.splits import make_splits
from blindspot.generators import claude_personas as cp
from blindspot.generators.claude_client import CostLimitExceeded, ResponseCache, Usage

MODEL = "claude-sonnet-5-5"
SENTINEL_PERSONA = "SENTINEL_PERSONA_TEXT_from_twin"
SENTINEL_ANSWER = "SENTINEL_ANSWER_9137"
ITEMS = [Item("q_a", "How happy are you?", "ordinal", ["very", "pretty", "not too"], "life"),
         Item("q_b", "Trust in science?", "ordinal", ["low", "mid", "high"], "science"),
         Item("q_c", "Own a car?", "binary", ["no", "yes"], "life")]


def _demo(n: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "age_band": rng.choice(["18-29", "65+"], n), "gender": rng.choice(["Female", "Male"], n),
        "education": "High school graduate", "income_band": "unknown", "region": "South", "race": "White",
        "weight": 0.7, "persona_text": SENTINEL_PERSONA, "answers_json": SENTINEL_ANSWER,
    }, index=[str(i) for i in range(n)])


def _good_json(items=ITEMS, skip: frozenset = frozenset()) -> str:
    out = []
    for n, it in enumerate(items, start=1):
        if it.code in skip:
            continue
        k = len(it.options)
        out.append({"q": n, "p": [100 // k] * (k - 1) + [100 - (100 // k) * (k - 1)], "a": k - 1})
    return json.dumps({"answers": out})


class FakeClient:
    """Stands in for anthropic.Anthropic; records every paid call."""

    def __init__(self, responder=lambda params, n: _good_json(), input_tokens: int = 500):
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


@pytest.fixture
def cache(tmp_path):
    return ResponseCache(tmp_path / "cache")


def test_prompts_never_contain_persona_text_or_answer_data():
    plan = cp.plan("toy", _demo(), ITEMS, MODEL)
    blob = json.dumps([s.params for s in plan.specs])
    assert SENTINEL_PERSONA not in blob and SENTINEL_ANSWER not in blob
    assert "0.7" not in json.dumps([s.params["messages"] for s in plan.specs])   # weight column ignored
    assert "unknown" not in plan.personas[0].lower()
    assert "18-29" in blob or "65+" in blob


def test_persona_uses_only_allowed_fields():
    row = pd.Series({"gender": "female", "age_band": "30-49", "region": "west", "persona_text": SENTINEL_PERSONA})
    text = cp.persona_text(row)
    assert "female" in text and "30-49" in text and "west" in text and SENTINEL_PERSONA not in text


def test_request_uses_structured_output_and_no_sampling_fields():
    params = cp.plan("toy", _demo(1), ITEMS, MODEL).specs[0].params
    assert params["output_config"]["format"]["type"] == "json_schema"
    assert "temperature" not in params and "tool_choice" not in params
    item_schema = params["output_config"]["format"]["schema"]["properties"]["answers"]["items"]
    assert set(item_schema["required"]) == {"q", "p", "a"}


def test_parsing_normalizes_and_rejects_bad_values():
    rec = {"text": json.dumps({"answers": [{"q": 1, "p": [20, 30, 30], "a": 1}, {"q": 2, "p": [-5, 50, 55], "a": 0},
                                           {"q": 3, "p": [1, 2, 3], "a": 0}]}), "stop_reason": "end_turn"}
    parsed = cp.parse_record(rec, ITEMS)
    assert parsed.probs["q_a"].sum() == pytest.approx(1.0) and parsed.probs["q_a"][0] == pytest.approx(0.25)
    assert "q_b" not in parsed.probs and any("negative" in p for p in parsed.problems)
    assert "q_c" not in parsed.probs  # three values for a two-option item
    assert not parsed.complete(ITEMS)
    assert cp.parse_record({"text": "not json", "stop_reason": "end_turn"}, ITEMS).problems == ["invalid JSON"]


def test_missing_item_is_retried_once_then_nan(cache):
    always_missing = FakeClient(lambda params, n: _good_json(skip=frozenset({"q_b"})))
    p = cp.plan("toy", _demo(2), ITEMS, MODEL)
    crowds = cp.execute(p, always_missing, cache, Usage(), batch=False)
    assert len(always_missing.calls) == 4                     # 2 people + 1 retry each
    assert crowds.mode.answers["q_b"].isna().all() and crowds.sampled.answers["q_b"].isna().all()
    assert crowds.mode.answers["q_a"].notna().all()
    assert set(crowds.failures) == {"person_00000", "person_00001"}

    fixed_on_retry = FakeClient(lambda params, n: _good_json(skip=frozenset({"q_b"}) if n <= 2 else frozenset()))
    crowds2 = cp.execute(cp.plan("toy2", _demo(2), ITEMS, MODEL), fixed_on_retry,
                         ResponseCache(cache.dir / "b"), Usage(), batch=False)
    assert crowds2.mode.answers.notna().all().all() and not crowds2.failures


def test_sampling_is_deterministic_for_a_fixed_seed():
    probs = {"q_a": np.array([0.2, 0.5, 0.3]), "q_b": np.array([0.1, 0.1, 0.8]), "q_c": np.array([0.5, 0.5])}
    draws = [cp.sample_answers(probs, ITEMS, seed=42, person=i) for i in range(50)]
    again = [cp.sample_answers(probs, ITEMS, seed=42, person=i) for i in range(50)]
    other = [cp.sample_answers(probs, ITEMS, seed=7, person=i) for i in range(50)]
    assert draws == again and draws != other


def test_cache_hit_makes_zero_api_calls(cache):
    client = FakeClient()
    p = cp.plan("toy", _demo(4), ITEMS, MODEL)
    first = cp.execute(p, client, cache, Usage(), batch=True)
    n_calls = len(client.calls)
    assert n_calls == 4
    usage = Usage()
    second = cp.execute(p, client, cache, usage, batch=True)
    assert len(client.calls) == n_calls and usage.calls == 0 and usage.cost_usd == 0
    assert first.mode.answers.equals(second.mode.answers)
    assert cp.estimate(p, client, cache)["calls"] == 0


def test_identical_personas_get_independent_calls():
    demo = pd.DataFrame({"age_band": ["30-49"] * 3, "gender": ["Male"] * 3}, index=["a", "b", "c"])
    keys = [s.key for s in cp.plan("toy", demo, ITEMS, MODEL).specs]
    assert len(set(keys)) == 3


def test_cost_guard_stops_above_threshold_without_calling(cache):
    client = FakeClient(input_tokens=2_000_000)        # absurd prompt size -> huge estimate
    p = cp.plan("toy", _demo(6), ITEMS, MODEL)
    with pytest.raises(CostLimitExceeded):
        guard({"toy": p}, client, cache, limit=5.0, batch=True)
    assert client.calls == []


def test_bunching_share():
    items = ITEMS[:2]
    ans = pd.DataFrame({"q_a": [0] * 9 + [1], "q_b": [0, 1] * 5}, index=list("abcdefghij"), dtype=float)
    r = Respondents(ans, pd.DataFrame(index=ans.index), items, "t")
    assert bunching_share(r) == 0.5


def test_experiment1_runs_end_to_end_with_fake_client(tmp_path):
    rng = np.random.default_rng(3)
    n = 120
    ids = [str(i) for i in range(n)]
    ans = pd.DataFrame(rng.integers(0, 2, size=(n, 3)), index=ids, columns=[i.code for i in ITEMS], dtype=float)
    real = Respondents(ans, _demo(n).set_axis(ids), ITEMS, "toy")
    splits = make_splits(ids, seed=1)
    keep = splits["pool"] + splits["test"]
    d = DatasetInput("twin2k", "Toy twin", real.subset(keep), splits,
                     fixed_crowds={"gpt41mini_twin": real.subset(splits["test"])})
    run_experiment0([d], out_dir=tmp_path, n_boot=10, maps=False)
    client = FakeClient(lambda params, k: _good_json() if k % 2 else _good_json(skip=frozenset({"q_c"})))
    report = run_experiment1([d], client, out_dir=tmp_path, cache=ResponseCache(tmp_path / "cache"), model=MODEL,
                             n_boot=10, maps=False, h2_path=tmp_path / "experiment0.json",
                             prompt_dir=tmp_path / "prompts")
    crowds = report["datasets"]["twin2k"]["crowds"]
    assert {"claude_mode", "claude_sampled", "gpt41mini_twin", "gaussian_copula"} <= set(crowds)
    assert report["robustness_twin"]["ranking_changed"] in (True, False)
    assert report["cost"]["actual"]["calls"] >= len(splits["test"])
    md = (tmp_path / "experiment1.md").read_text(encoding="utf-8")
    assert "**claude_mode**" in md and "Robustness" in md and "Cost:" in md
    prompts = (tmp_path / "prompts" / "twin2k_empty_p1_user_prompts.txt").read_text("utf-8")
    assert SENTINEL_PERSONA not in prompts and SENTINEL_ANSWER not in prompts


def test_interrupted_batch_is_resumed_not_resubmitted(cache):
    from blindspot.generators.claude_client import pending_batch_path, run_batch

    specs = cp.plan("toy", _demo(3), ITEMS, MODEL).specs
    finished = FakeClient()
    results = [SimpleNamespace(custom_id=s.key, result=SimpleNamespace(type="succeeded",
                                                                       message=finished._message(dict(s.params))))
               for s in specs]

    def no_create(**_):
        raise AssertionError("must not resubmit a pending batch")

    client = SimpleNamespace(messages=SimpleNamespace(batches=SimpleNamespace(
        create=no_create, retrieve=lambda _id: SimpleNamespace(id=_id, processing_status="ended"),
        results=lambda _id: results)))
    path = pending_batch_path(specs, cache.dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"id": "msgbatch_old", "n": len(specs)}), encoding="utf-8")
    out = run_batch(client, specs, log=lambda _: None, cache_dir=cache.dir)
    assert set(out) == {s.key for s in specs} and not path.exists()
