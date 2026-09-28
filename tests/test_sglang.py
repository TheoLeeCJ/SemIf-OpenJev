"""SGLang backend tests against a standard library fake server: no model downloads, GPU, or SGLang needed."""

import hashlib
import http.server
import json
import math
import os
import re
import socket
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from semif_phase1 import reranker, sglang_backend
from semif_phase1.core import LETTERS, softmax
from semif_phase1.direct import encode_prompt
from semif_phase1.shared import _state_prefix

SOURCE, REVISION = "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
RERANKER_SOURCE, RERANKER_REVISION = "Qwen/Qwen3-Reranker-4B", "22e683669bc0f0bd69640a1354a6d0aebcfeede5"


class Tokenizer:
    """Byte tokenizer with single-token yes and no, as the reranker contract needs."""

    words = {"no": 300, "yes": 301}

    def apply_chat_template(self, turns, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        assert tokenize is False and add_generation_prompt is True and enable_thinking is False
        return "".join(f"<{turn['role']}>{turn['content']}\n" for turn in turns) + "<assistant>"

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [self.words[text]] if text in self.words else list(text.encode())

    def decode(self, ids):
        return bytes(ids).decode()

    def convert_tokens_to_ids(self, text):
        return self.words[text]


def logprob(item, token):
    """Deterministic fake log-probability of one candidate after one item."""
    return -((token % 5) + 1) / 4 - len(item) / 1e4


class FakeSGLang(http.server.ThreadingHTTPServer):
    def __init__(self):
        super().__init__(("127.0.0.1", 0), Handler)
        self.tokenizer, self.requests = Tokenizer(), []
        self.model_info = {"model_path": SOURCE, "is_generation": True, "preferred_sampling_params": None,
                           "load_format": "auto"}
        self.server_info = {"model_path": SOURCE, "revision": REVISION, "max_req_input_len": 8192, "enable_mis": False,
                            "allow_auto_truncate": False, "api_key": "server-secret", "admin_api_key": "admin-secret",
                            "launch_command": "--api-key server-secret"}
        self.routes = {"/model_info": lambda body: (200, self.model_info),
                       "/server_info": lambda body: (200, self.server_info),
                       "/v1/tokenize": self.tokenize, "/v1/score": self.score}

    def tokenize(self, body):
        return 200, {"tokens": [self.tokenizer.encode(text, add_special_tokens=False) for text in body["prompt"]]}

    def score(self, body):
        values = [[logprob(item, token) for token in candidates]
                  for item, candidates in zip(body["items"], body["label_token_ids"])]
        return 200, {"token_logprobs": values, "usage": {"prompt_tokens": sum(map(len, body["items"]))}}

    def scores(self):
        return [request for request in self.requests if request["path"] == "/v1/score"]


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.answer(None)

    def do_POST(self):
        self.answer(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))

    def answer(self, body):
        self.server.requests.append({"method": self.command, "path": self.path, "headers": dict(self.headers),
                                     "body": body})
        status, payload = self.server.routes[self.path](body)
        if status is None:
            return self.wfile.write(payload or b"")  # Raw bytes, or close without an answer.
        data = (payload if isinstance(payload, str) else json.dumps(payload)).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def server(monkeypatch):
    fake, tokenizer = FakeSGLang(), Tokenizer()
    threading.Thread(target=fake.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
    fake.url, fake.loads = f"http://127.0.0.1:{fake.server_address[1]}", []
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(__version__="test", AutoTokenizer=SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: fake.loads.append((args, kwargs)) or tokenizer)))
    monkeypatch.delenv(sglang_backend.API_KEY_ENV, raising=False)
    fake.load = lambda **kwargs: sglang_backend.load_model(
        kwargs.pop("source", SOURCE), kwargs.pop("revision", REVISION), url=kwargs.pop("url", fake.url), **kwargs)
    yield fake
    fake.shutdown()
    fake.server_close()


def row(identifier, state="shared evidence", options=2, question="Which answer follows?"):
    return {"id": identifier, "state": state, "question": question,
            "options": [{"id": f"o{index}", "description": f"Option {index}"} for index in range(options)]}


def closed_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_load_model_verifies_server_and_records_only_allowlisted_facts(server, monkeypatch):
    monkeypatch.setenv(sglang_backend.API_KEY_ENV, "client-key")
    model, tokenizer, metadata = server.load(url=server.url + "/", max_tokens=8191)
    assert [(request["method"], request["path"]) for request in server.requests] == [
        ("GET", "/model_info"), ("GET", "/server_info"), ("POST", "/v1/tokenize"), ("POST", "/v1/score")]
    assert all(request["headers"]["Authorization"] == "Bearer client-key"
               and request["headers"]["Content-Type"] == "application/json" for request in server.requests)
    assert server.loads == [((SOURCE,), {"revision": REVISION, "local_files_only": False, "trust_remote_code": False})]
    tokenize, probe = server.requests[2]["body"], server.requests[3]["body"]
    assert tokenize["add_special_tokens"] is False and tokenize["prompt"][1:] == [*LETTERS, "yes", "no"]
    assert probe == {"query": [], "items": [tokenizer.encode(tokenize["prompt"][0])],
                     "label_token_ids": [[ord(letter) for letter in LETTERS]], "return_token_logprobs": True}
    assert metadata["backend"] == "sglang" and metadata["server_url"] == server.url
    assert metadata["max_prompt_tokens"] == 8191 and metadata["server"]["revision"] == REVISION
    assert metadata["server"]["load_format"] == "auto"
    assert not any(secret in json.dumps(metadata) for secret in ("server-secret", "admin-secret", "client-key"))


def test_local_source_accepts_its_label_without_a_served_revision(server, tmp_path):
    server.model_info["model_path"] = server.server_info["model_path"] = str(tmp_path)
    server.server_info["revision"] = None
    _, _, metadata = server.load(source=str(tmp_path), revision="local-checkpoint-v1")
    assert server.loads[0][1] == {"revision": None, "local_files_only": True, "trust_remote_code": False}
    assert metadata["revision"] == "local-checkpoint-v1" and metadata["server"]["revision"] is None


@pytest.mark.parametrize("change,message", [
    (lambda fake: fake.model_info.update(model_path=RERANKER_SOURCE), "Pass --sglang-url"),
    (lambda fake: fake.model_info.update(is_generation=False), "non-generation model. Relaunch it without"),
    (lambda fake: fake.model_info.update(load_format="dummy"), "--load-format dummy, which serves random weights"),
    (lambda fake: fake.server_info.update(revision="0" * 40), f"but --revision is {REVISION}. Relaunch it with"),
    (lambda fake: fake.server_info.update(revision=None), f"serves revision None, but --revision is {REVISION}"),
    (lambda fake: fake.server_info.update(enable_mis=True), "--enable-mis"),
    (lambda fake: fake.server_info.update(allow_auto_truncate=True), "--allow-auto-truncate"),
    (lambda fake: fake.model_info.update(preferred_sampling_params={"temperature": 0.5}),
     '--preferred-sampling-params {"temperature": 0.5}, which applies to every scoring request'),
    (lambda fake: fake.server_info.update(max_req_input_len=4096), "Use a --max-tokens below 4096"),
    (lambda fake: fake.server_info.pop("max_req_input_len"), "refuses inputs of None tokens"),
    (lambda fake: fake.routes.update({"/model_info": lambda body: (401, {"error": "Unauthorized"})}),
     "(HTTP 401). Check --sglang-url, and set SEMIF_SGLANG_API_KEY"),
    (lambda fake: fake.routes.update({"/server_info": lambda body: (200, "<html>proxy</html>")}),
     "did not return SGLang server information (HTTP 200)"),
    (lambda fake: fake.routes.update({"/v1/tokenize": lambda body: (400, "tokenizer unavailable")}),
     "(HTTP 400: tokenizer unavailable). Serve the same checkpoint"),
    (lambda fake: setattr(fake.tokenizer, "words", {"no": 300, "yes": 302}), "(different token ids)"),
    (lambda fake: setattr(fake.tokenizer, "words", {"no": 300, "yes": 301, "C": 99}), "(different token ids)"),
    (lambda fake: fake.routes.update({"/v1/tokenize": lambda body: (200, {"tokens": [[1], *[[65]] * 18]})}),
     "(different token ids)"),
    (lambda fake: fake.routes.update({"/v1/score": lambda body: (400, {"message": "label_token_ids: int_type"})}),
     "HTTP 400: label_token_ids: int_type. The server needs per-item candidate scoring"),
    (lambda fake: fake.routes.update({"/v1/score": lambda body: (200, {"scores": [[0.5] * 16]})}),
     "no token_logprobs list with one entry per item for the load-time score probe. The server needs"),
])
def test_load_model_refuses_unsupported_servers_with_the_fix(server, change, message):
    change(server)
    with pytest.raises(RuntimeError, match=re.escape(message)):
        server.load()


def test_unreachable_server_names_the_url_and_the_launch_before_loading_the_tokenizer(server):
    port = closed_port()
    url = f"http://127.0.0.1:{port}"
    with pytest.raises(RuntimeError) as error:
        server.load(url=url)
    assert f"Cannot reach an SGLang server at {url}" in str(error.value)
    assert (f"python -m sglang.launch_server --model-path {SOURCE} --revision {REVISION} --port {port}"
            in str(error.value))
    assert "fired up and ready to roll" in str(error.value)
    assert server.loads == []


@pytest.mark.parametrize("options,message", [
    ({"revision": "main"}, "pinned 40-character revision"),
    ({"source": ".", "revision": ""}, "Local sources require a revision label"),
    ({"url": "http://user:secret@127.0.0.1:30000"}, "without user info"),
    ({"url": "localhost:30000"}, "must be an http or https URL"),
    ({"url": "http://127.0.0.1:30000/?key=secret"}, "without user info, query, or fragment"),
    ({"url": "http://127.0.0.1:30000/#secret"}, "without user info, query, or fragment"),
])
def test_load_model_rejects_invalid_arguments_before_any_request(server, options, message):
    with pytest.raises(ValueError, match=re.escape(message)) as error:
        server.load(**options)
    assert "secret" not in str(error.value) and server.requests == []


def expected_row(tokenizer, source_row, max_tokens=4096):
    ids, slots, prompt_hash = encode_prompt(tokenizer, source_row, max_tokens)
    values = [logprob(ids, token) for token in slots]
    return ids, slots, {
        "id": source_row["id"],
        "option_ids": [option["id"] for option in source_row["options"]],
        "probabilities": softmax(values),
        "option_logits": values,
        "answer_token_ids": slots,
        "input_tokens": len(ids),
        "allowed_token_mass": sum(math.exp(value) for value in values),
        "prompt_sha256": prompt_hash,
        "prompt_version": "direct-options-v1",
        "readout": sglang_backend.READOUT,
        "probability_status": "conditional option score, uncalibrated as decision confidence",
    }


def test_direct_serial_and_shared_send_semif_ids_and_build_rows_from_the_response(server):
    model, tokenizer, metadata = server.load()
    rows = [row("first"), row("second", question="Does the second answer follow?", options=3)]
    before = len(server.scores())
    direct = [sglang_backend.score(model, tokenizer, source_row, metadata) for source_row in rows]
    scorer = sglang_backend.SerialPrefixScorer(model, tokenizer, metadata)
    serial = [scorer.score(source_row) for source_row in rows]
    shared, timing = sglang_backend.score_shared(model, tokenizer, rows, metadata)
    requests = [request["body"] for request in server.scores()[before:]]
    expected = [expected_row(tokenizer, source_row) for source_row in rows]
    per_row = [{"query": [], "items": [ids], "label_token_ids": [slots], "return_token_logprobs": True}
               for ids, slots, _ in expected]
    assert requests == per_row + per_row + [{"query": [], "items": [ids for ids, _, _ in expected],
                                             "label_token_ids": [slots for _, slots, _ in expected],
                                             "return_token_logprobs": True}]
    prefix = _state_prefix(tokenizer, "shared evidence")
    for mode, results in (("direct", direct), ("serial", serial), ("shared", shared)):
        for result, (_, _, fields) in zip(results, expected):
            assert {key: result[key] for key in fields} == fields
            assert result["model"] == {**metadata, "serving_config": f"sglang-score-{mode}-v1"}
            assert not {"cache_hit", "forward_seconds", "full_vocab_argmax_id", "prefill_seconds"} & result.keys()
            assert ("request_seconds" in result) == ("total_seconds" in result) == (mode != "shared")
            assert ("prefix_sha256" in result) == (mode == "serial")
    assert [result["prefix_tokens"] for result in serial] == [len(prefix)] * 2
    assert serial[0]["prefix_sha256"] == hashlib.sha256(json.dumps(prefix).encode()).hexdigest()
    assert timing["batch_size"] == 2 and timing["prefix_tokens"] == len(prefix)
    assert timing["true_suffix_tokens"] == sum(len(ids) - len(prefix) for ids, _, _ in expected)


def test_reranker_sends_one_item_per_option_with_no_and_yes_candidates(server):
    model, tokenizer, metadata = server.load()
    source_row = row("pairs", options=3)
    result = sglang_backend.reranker_score(model, tokenizer, source_row, metadata)
    pairs = [reranker._encode(tokenizer, source_row, option, 4096) for option in source_row["options"]]
    assert server.scores()[-1]["body"] == {"query": [], "items": [ids for ids, _ in pairs],
                                           "label_token_ids": [[300, 301]] * 3, "return_token_logprobs": True}
    values = [[logprob(ids, 300), logprob(ids, 301)] for ids, _ in pairs]
    odds = [yes - no for no, yes in values]
    assert result["option_logits"] == odds and result["probabilities"] == softmax(odds)
    assert result["independent_binary_relevance"] == [softmax(pair)[1] for pair in values]
    assert result["input_tokens"] == sum(len(ids) for ids, _ in pairs)
    assert result["max_option_input_tokens"] == max(len(ids) for ids, _ in pairs)
    assert result["option_prompt_sha256"] == [prompt_hash for _, prompt_hash in pairs]
    assert result["prompt_version"] == "qwen3-reranker-native-options-v1"
    assert result["model"]["serving_config"] == "sglang-score-reranker-v1"
    assert result["probability_status"] == "relative option compatibility, uncalibrated as categorical probability"
    with pytest.raises(ValueError, match="Option IDs must be unique"):
        sglang_backend.reranker_score(model, tokenizer, {**source_row, "options": [source_row["options"][0]] * 2},
                                      metadata)


def reshape(fake, change):
    def answer(body):
        status, payload = fake.score(body)
        change(payload)
        return status, payload
    return answer


@pytest.mark.parametrize("route,message", [
    (lambda fake: lambda body: (400, {"message": "Token ID 99 is out of vocabulary"}),
     "rejected row first with HTTP 400: Token ID 99 is out of vocabulary"),
    (lambda fake: lambda body: time.sleep(1) or fake.score(body), "did not answer row first within 0.2 s"),
    (lambda fake: lambda body: (None, None), "for row first (RemoteDisconnected"),
    (lambda fake: lambda body: (None, b"HTTP/1.1 200 OK\r\nContent-Length: 9\r\n\r\n{"),
     "for row first (IncompleteRead"),
    (lambda fake: reshape(fake, lambda payload: payload["usage"].update(prompt_tokens=3)),
     "counted 3 prompt tokens for row first"),
    (lambda fake: reshape(fake, lambda payload: payload.pop("usage")), "counted None prompt tokens for row first"),
    (lambda fake: reshape(fake, lambda payload: payload["token_logprobs"][0].__setitem__(1, None)),
     "for row first, expected 2 finite values"),
    (lambda fake: reshape(fake, lambda payload: payload["token_logprobs"][0].__setitem__(1, math.inf)),
     "for row first, expected 2 finite values"),
    (lambda fake: reshape(fake, lambda payload: payload["token_logprobs"][0].pop()),
     "for row first, expected 2 finite values"),
    (lambda fake: reshape(fake, lambda payload: payload["token_logprobs"].__setitem__(0, None)),
     "token_logprobs None for row first"),
    (lambda fake: reshape(fake, lambda payload: payload.pop("token_logprobs")), "no token_logprobs list"),
])
def test_runtime_failures_stop_with_the_row_and_are_never_retried(server, route, message):
    model, tokenizer, metadata = server.load(timeout=0.2)
    server.routes["/v1/score"] = route(server)
    before = len(server.scores())
    with pytest.raises(RuntimeError, match=re.escape(message)):
        sglang_backend.score(model, tokenizer, row("first"), metadata)
    assert len(server.scores()) == before + 1


@pytest.mark.parametrize("mode", ["shared", "reranker"])
def test_a_missing_item_stops_requests_that_carry_several_items(server, mode):
    model, tokenizer, metadata = server.load()
    server.routes["/v1/score"] = reshape(server, lambda payload: payload["token_logprobs"].pop())
    with pytest.raises(RuntimeError, match="no token_logprobs list with one entry per item"):
        if mode == "shared":
            sglang_backend.score_shared(model, tokenizer, [row("first"), row("second"), row("third")], metadata)
        else:
            sglang_backend.reranker_score(model, tokenizer, row("pairs", options=3), metadata)


def test_boundaries_of_options_states_text_and_input_limit(server):
    model, tokenizer, metadata = server.load()
    rows = [
        row("sixteen", state={"zone": "Zürich", "events": ["部署完成", 3]}, options=16),
        row("array", state=[{"check": "passed"}, "done"]),
        row("text", state="Évidence non-ASCII: 東京"),
    ]
    results = [sglang_backend.score(model, tokenizer, source_row, metadata) for source_row in rows]
    assert [len(request["body"]["label_token_ids"][0]) for request in server.scores()[-3:]] == [16, 2, 2]
    assert [len(result["probabilities"]) for result in results] == [16, 2, 2]
    exact = len(encode_prompt(tokenizer, rows[2], 4096)[0])
    before = len(server.scores())
    sglang_backend.score(model, tokenizer, rows[2], metadata, max_tokens=exact)
    with pytest.raises(ValueError, match="no truncation"):
        sglang_backend.score(model, tokenizer, rows[2], metadata, max_tokens=exact - 1)
    assert len(server.scores()) == before + 1


def test_serial_accepts_interleaved_and_reordered_states_by_their_own_prefix(server):
    model, tokenizer, metadata = server.load()
    rows = [row("a1", state="alpha"), row("b1", state="beta"), row("a2", state="alpha"),
            row("d1", state={"x": 1, "y": True}), row("d2", state={"y": True, "x": 1})]
    scorer = sglang_backend.SerialPrefixScorer(model, tokenizer, metadata)
    results = [scorer.score(source_row) for source_row in rows]
    prefixes = [_state_prefix(tokenizer, source_row["state"]) for source_row in rows]
    assert prefixes[3] != prefixes[4]
    assert [result["prefix_tokens"] for result in results] == [len(prefix) for prefix in prefixes]
    assert [result["prefix_sha256"] for result in results] == [
        hashlib.sha256(json.dumps(prefix).encode()).hexdigest() for prefix in prefixes]
    assert [request["body"]["items"][0] for request in server.scores()[-5:]] == [
        encode_prompt(tokenizer, source_row, 4096)[0] for source_row in rows]


@pytest.mark.parametrize("mode,rows,message", [
    ("serial", [row("first")], "--mode direct sends the same SGLang requests"),
    ("shared", [row("first"), row("second")], "--mode direct scores these rows"),
    ("shared", [row("first"), row("second", state="other evidence")], "one nonempty exact state"),
    ("shared", [row("first"), row("first")], "Decision IDs must be unique"),
])
def test_serial_and_shared_input_rules_refuse_before_any_request(server, monkeypatch, mode, rows, message):
    model, tokenizer, metadata = server.load()
    monkeypatch.setattr(sglang_backend, "_state_prefix", lambda tokenizer, state: [0])
    before = len(server.scores())
    with pytest.raises(ValueError, match=message):
        if mode == "serial":
            sglang_backend.SerialPrefixScorer(model, tokenizer, metadata).score(rows[0])
        else:
            sglang_backend.score_shared(model, tokenizer, rows, metadata)
    assert len(server.scores()) == before


REAL_ROWS = [
    {"id": key, "state": "The deployment completed at 14:02 UTC. Health checks passed in all three zones.",
     "question": question, "options": [{"id": "yes", "description": yes}, {"id": "no", "description": no}]}
    for key, question, yes, no in [
        ("deployment", "Is there evidence that the deployment succeeded?", "The deployment succeeded.",
         "The deployment did not succeed."),
        ("health", "Did the health checks pass?", "The checks passed.", "The checks failed."),
    ]
]


def assert_valid_distributions(results):
    assert [result["id"] for result in results] == [source_row["id"] for source_row in REAL_ROWS]
    for result in results:
        assert all(math.isfinite(value) for value in result["probabilities"])
        assert math.isclose(sum(result["probabilities"]), 1.0, rel_tol=1e-9)


@pytest.mark.skipif(not os.environ.get("SEMIF_SGLANG_URL"),
                    reason="set SEMIF_SGLANG_URL to a server for Qwen/Qwen3.5-4B at the pinned revision")
def test_real_server_scores_direct_serial_and_shared():
    model, tokenizer, metadata = sglang_backend.load_model(SOURCE, REVISION, url=os.environ["SEMIF_SGLANG_URL"])
    direct = [sglang_backend.score(model, tokenizer, source_row, metadata) for source_row in REAL_ROWS]
    scorer = sglang_backend.SerialPrefixScorer(model, tokenizer, metadata)
    serial = [scorer.score(source_row) for source_row in REAL_ROWS]
    shared, _ = sglang_backend.score_shared(model, tokenizer, REAL_ROWS, metadata)
    for results in (direct, serial, shared):
        assert_valid_distributions(results)
        for result, source_row in zip(results, REAL_ROWS):
            ids, slots, prompt_hash = encode_prompt(tokenizer, source_row, 4096)
            assert (result["prompt_sha256"], result["input_tokens"], result["answer_token_ids"]) == (
                prompt_hash, len(ids), slots)
        # Values move slightly between cold and cached requests on hybrid models, so compare choices only.
        assert [result["option_ids"][result["probabilities"].index(max(result["probabilities"]))]
                for result in results] == ["yes", "yes"]
    assert metadata["server"]["revision"] == REVISION


@pytest.mark.skipif(not os.environ.get("SEMIF_SGLANG_RERANKER_URL"),
                    reason="set SEMIF_SGLANG_RERANKER_URL to a server for Qwen/Qwen3-Reranker-4B "
                           "at the pinned revision")
def test_real_server_scores_reranker_pairs():
    model, tokenizer, metadata = sglang_backend.load_model(
        RERANKER_SOURCE, RERANKER_REVISION, url=os.environ["SEMIF_SGLANG_RERANKER_URL"])
    results = [sglang_backend.reranker_score(model, tokenizer, source_row, metadata) for source_row in REAL_ROWS]
    assert_valid_distributions(results)
    for result, source_row in zip(results, REAL_ROWS):
        pairs = [reranker._encode(tokenizer, source_row, option, 4096) for option in source_row["options"]]
        assert result["option_prompt_sha256"] == [prompt_hash for _, prompt_hash in pairs]
        assert result["input_tokens"] == sum(len(ids) for ids, _ in pairs)
    assert metadata["server"]["revision"] == RERANKER_REVISION
