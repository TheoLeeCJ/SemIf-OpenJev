from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from semif_phase1.shared import _suffix_layout, score_shared  # noqa: E402


def test_suffix_padding_follows_real_tokens():
    layout, ends = _suffix_layout([[3, 4], [5]], 7, 0)
    assert layout["input_ids"] == [[3, 4], [5, 0]]
    assert layout["attention_mask"] == [[1] * 9, [1] * 8 + [0]]
    assert layout["position_ids"] == [[7, 8], [7, 0]]
    assert ends == [1, 0]


def test_empty_suffix_is_rejected():
    with pytest.raises(ValueError):
        _suffix_layout([[1], []], 7, 0)


class FakeCache:
    """Native-cache stand-in: mutable length, records reorder calls."""

    def __init__(self, length):
        self.length = length
        self.orders = []

    def get_seq_length(self):
        return self.length

    def reorder_cache(self, order):
        self.orders.append(order.tolist() if hasattr(order, "tolist") else list(order))

    def __deepcopy__(self, memo):
        clone = FakeCache(self.length)
        memo[id(self)] = clone
        return clone


class FakeModel:
    """Forward with a real signature so selective-logits detection succeeds."""

    def __init__(self, calls, device):
        self.calls = calls
        self._device = device

    def parameters(self):
        yield SimpleNamespace(device=self._device)

    def eval(self):
        return self

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, use_cache=False,
                return_dict=True, logits_to_keep=1, past_key_values=None):
        batch = int(input_ids.shape[0])
        width = int(input_ids.shape[1])
        entry_length = None if past_key_values is None else past_key_values.length
        self.calls.append({
            "batch": batch,
            "entry_length": entry_length,
            "reorder_calls": [] if past_key_values is None else list(past_key_values.orders),
            "logits_to_keep": (
                logits_to_keep if isinstance(logits_to_keep, int) else logits_to_keep.tolist()
            ),
        })
        if past_key_values is None:
            past_key_values = FakeCache(width)
        else:
            past_key_values.length += width
        positions = 1 if isinstance(logits_to_keep, int) else len(logits_to_keep)
        vocabulary = torch.arange(width + 4, dtype=torch.float32).reshape(1, 1, -1)
        row_offsets = torch.arange(batch, dtype=torch.float32).reshape(-1, 1, 1) * 100
        position_offsets = torch.arange(positions, dtype=torch.float32).reshape(1, -1, 1) * 10
        logits = vocabulary + row_offsets + position_offsets
        return SimpleNamespace(logits=logits, past_key_values=past_key_values)

    __call__ = forward


def test_score_shared_looped_path_forwards_one_row_per_call(monkeypatch):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    rows_data = [
        {"id": key, "state": "s", "question": f"q{index}",
         "options": [{"id": "x", "description": "X"}, {"id": "y", "description": "Y"}]}
        for index, key in enumerate(("a", "b"))
    ]
    monkeypatch.setattr("semif_phase1.shared.encode_prompt",
                        lambda tok, row, limit: ([1] * 8, [2, 3], f"hash-{row['id']}"))
    monkeypatch.setattr("semif_phase1.shared._state_prefix", lambda tokenizer, state: [1] * 6)
    calls = []
    results, timing = score_shared(FakeModel(calls, torch.device("mps")),
                                   SimpleNamespace(pad_token_id=0, eos_token_id=1), rows_data, {}, 4096)
    assert [r["id"] for r in results] == ["a", "b"]
    assert len(calls) == 3  # one prefill + one batch-1 suffix forward per row
    prefill, *suffixes = calls
    assert prefill["entry_length"] is None
    # Each branch starts at prefix length: deep-copied, not shared state.
    assert all(call["batch"] == 1 and call["entry_length"] == 6 for call in suffixes)
    assert timing["replicate_seconds"] > 0 and timing["suffix_forward_seconds"] > 0
    assert timing["padded_suffix_tokens"] == timing["true_suffix_tokens"] == 4
    assert all(sum(result["probabilities"]) == pytest.approx(1.0) for result in results)


def test_score_shared_batched_path_prefills_once_and_preserves_order(monkeypatch):
    rows_data = [
        {"id": key, "state": "shared", "question": f"q{index}",
         "options": [{"id": "x", "description": "X"}, {"id": "y", "description": "Y"}]}
        for index, key in enumerate(("first", "second", "third"))
    ]
    suffixes = {"first": [7, 8, 9], "second": [10], "third": [11, 12, 13]}
    monkeypatch.setattr(
        "semif_phase1.shared.encode_prompt",
        lambda tok, row, limit: ([1] * 6 + suffixes[row["id"]], [2, 3], f"hash-{row['id']}"),
    )
    monkeypatch.setattr("semif_phase1.shared._state_prefix", lambda tokenizer, state: [1] * 6)
    calls = []
    results, timing = score_shared(
        FakeModel(calls, torch.device("cpu")),
        SimpleNamespace(pad_token_id=0, eos_token_id=1),
        rows_data,
        {},
        4096,
    )

    assert [result["id"] for result in results] == ["first", "second", "third"]
    assert len(calls) == 2
    prefill, suffix_batch = calls
    assert prefill == {
        "batch": 1,
        "entry_length": None,
        "reorder_calls": [],
        "logits_to_keep": 1,
    }
    assert suffix_batch["batch"] == 3
    assert suffix_batch["entry_length"] == 6
    assert suffix_batch["reorder_calls"] == [[0, 0, 0]]
    assert suffix_batch["logits_to_keep"] == [0, 2]
    assert timing["batch_size"] == 3
    assert timing["prefix_tokens"] == 6
    assert timing["true_suffix_tokens"] == 7
    assert timing["padded_suffix_tokens"] == 9
    assert all(result["model"]["serving_config"] == "native-state-prefix-parallel-v1" for result in results)
    assert [result["option_logits"] for result in results] == [[12.0, 13.0], [102.0, 103.0], [212.0, 213.0]]
    assert all(sum(result["probabilities"]) == pytest.approx(1.0) for result in results)
