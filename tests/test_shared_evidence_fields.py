"""score_shared must keep the per-row evidence fields serial/direct predictions carry."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

torch = pytest.importorskip("torch")

from semif_phase1.shared import score_shared  # noqa: E402


class FakeCache:
    def __init__(self, length):
        self.length = length

    def get_seq_length(self):
        return self.length

    def reorder_cache(self, order):
        return None

    def __deepcopy__(self, memo):
        clone = FakeCache(self.length)
        memo[id(self)] = clone
        return clone


class FakeModel:
    def __init__(self, device):
        self._device = device

    def parameters(self):
        yield SimpleNamespace(device=self._device)

    def eval(self):
        return self

    def forward(self, input_ids=None, attention_mask=None, use_cache=False,
                return_dict=True, logits_to_keep=1, past_key_values=None, **kwargs):
        width = int(input_ids.shape[1])
        batch = int(input_ids.shape[0])
        if past_key_values is None:
            past_key_values = FakeCache(width)
        else:
            past_key_values.length += width
        # logits shape [batch, kept, vocab]; keep one position with vocab 8
        vocab = 8
        kept = 1 if isinstance(logits_to_keep, int) else int(logits_to_keep.numel())
        logits = torch.zeros(batch, kept, vocab, dtype=torch.float32)
        for i in range(batch):
            logits[i, -1, 2] = 1.0
            logits[i, -1, 3] = 2.0
        return SimpleNamespace(logits=logits, past_key_values=past_key_values)

    __call__ = forward


REQUIRED = {
    "allowed_token_mass",
    "answer_token_ids",
    "batch_size",
    "full_vocab_argmax_id",
    "padded_tokens",
}


def test_score_shared_emits_evidence_fields(monkeypatch):
    rows = [
        {"id": "a", "state": "s", "question": "q0",
         "options": [{"id": "x", "description": "X"}, {"id": "y", "description": "Y"}]},
        {"id": "b", "state": "s", "question": "q1",
         "options": [{"id": "x", "description": "X"}, {"id": "y", "description": "Y"}]},
    ]
    monkeypatch.setattr(
        "semif_phase1.shared.encode_prompt",
        lambda tok, row, limit: ([1] * 8, [2, 3], f"hash-{row['id']}"),
    )
    monkeypatch.setattr("semif_phase1.shared._state_prefix", lambda tokenizer, state: [1] * 6)
    results, timing = score_shared(
        FakeModel(torch.device("cpu")),
        SimpleNamespace(pad_token_id=0, eos_token_id=1),
        rows,
        {},
        4096,
    )
    assert [r["id"] for r in results] == ["a", "b"]
    for result in results:
        missing = REQUIRED - result.keys()
        assert not missing, f"missing {missing}"
        assert result["batch_size"] == 2
        assert result["answer_token_ids"] == [2, 3]
        assert result["full_vocab_argmax_id"] == 3
        assert 0.0 < result["allowed_token_mass"] <= 1.0
        assert result["padded_tokens"] >= 1
    assert timing["batch_size"] == 2
