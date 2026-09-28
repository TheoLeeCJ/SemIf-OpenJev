"""Option readout through the /v1/score endpoint of a running SGLang server.

Prompt construction, answer-slot verification, and input validation stay on the pinned reference tokenizer, so
prompt_sha256, input_tokens, and answer_token_ids match the Torch backend. SGLang only scores SemIf's own token ids
and returns the full-vocabulary log-probability of each declared answer slot, which needs sgl-project/sglang#40826.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import re
import time
import urllib.parse

from . import reranker
from .core import LETTERS, direct_messages, softmax, validate_row
from .direct import PROMPT_VERSION, _slot_ids, encode_prompt
from .shared import _state_prefix

DEFAULT_URL = "http://127.0.0.1:30000"
DEFAULT_TIMEOUT_SECONDS = 300.0
API_KEY_ENV = "SEMIF_SGLANG_API_KEY"
READOUT = ("SGLang /v1/score full-vocabulary next-position log-probabilities of the declared answer slots, "
           "one constant per row away from raw logits")
# An allowlist, because /server_info also returns the API keys and the launch command.
SERVER_FIELDS = ("model_path", "served_model_name", "architectures", "model_type", "weight_version", "load_format",
                 "version", "revision", "dtype", "quantization", "kv_cache_dtype", "tp_size", "dp_size", "pp_size",
                 "attention_backend", "prefill_attention_backend", "enable_fp32_lm_head", "page_size",
                 "disable_radix_cache", "chunked_prefill_size", "speculative_algorithm", "max_req_input_len",
                 "json_model_override_args")


class _Client:
    """JSON over one direct HTTP connection per request, with no proxy, no redirect, and no retry."""

    def __init__(self, url: str, timeout: float, start_hint: str):
        self.url, self.parts, self.timeout, self.start_hint = url, urllib.parse.urlsplit(url), timeout, start_hint
        self.headers = {"Content-Type": "application/json"}
        if os.environ.get(API_KEY_ENV):
            self.headers["Authorization"] = f"Bearer {os.environ[API_KEY_ENV]}"

    def send(self, path: str, body: dict | None, subject: str):
        """Return the HTTP status and the JSON body, or its text, of one GET or POST request."""
        kind = http.client.HTTPSConnection if self.parts.scheme == "https" else http.client.HTTPConnection
        connection = kind(self.parts.netloc, timeout=self.timeout)
        try:
            connection.request("GET" if body is None else "POST", self.parts.path.rstrip("/") + path,
                               None if body is None else json.dumps(body, allow_nan=False).encode(), self.headers)
            response = connection.getresponse()
            status, text = response.status, response.read().decode("utf-8", "replace")
        except TimeoutError as error:
            raise RuntimeError(f"SGLang did not answer {subject} within {self.timeout:g} s. "
                               "Raise --sglang-timeout or check the server.") from error
        except (OSError, http.client.HTTPException) as error:
            raise RuntimeError(f"Cannot reach an SGLang server at {self.url} for {subject} ({error!r}). "
                               f"{self.start_hint}") from error
        finally:
            connection.close()
        try:
            return status, json.loads(text)
        except ValueError:
            return status, text


def _message(payload) -> str:
    return str(payload.get("message", payload) if isinstance(payload, dict) else payload)


def _score(client: _Client, items, candidates, subject: str, names: list[str]):
    """Require one finite log-probability per candidate and the prompt token count SemIf sent."""
    started = time.perf_counter()
    body = {"query": [], "items": items, "label_token_ids": candidates, "return_token_logprobs": True}
    status, payload = client.send("/v1/score", body, subject)
    seconds = time.perf_counter() - started
    if status != 200:
        raise RuntimeError(f"SGLang rejected {subject} with HTTP {status}: {_message(payload)}")
    values = payload.get("token_logprobs") if isinstance(payload, dict) else None
    if not isinstance(values, list) or len(values) != len(items):
        raise RuntimeError(f"SGLang returned no token_logprobs list with one entry per item for {subject}")
    for name, row, expected in zip(names, values, candidates):
        if not (isinstance(row, list) and len(row) == len(expected)
                and all(type(value) in (int, float) and math.isfinite(value) for value in row)):
            raise RuntimeError(f"SGLang returned token_logprobs {row!r} for {name}, "
                               f"expected {len(expected)} finite values")
    usage, sent = payload.get("usage"), sum(map(len, items))
    counted = usage.get("prompt_tokens") if isinstance(usage, dict) else None
    if counted != sent:
        raise RuntimeError(f"SGLang counted {counted} prompt tokens for {subject}, but SemIf sent {sent}")
    return [[float(value) for value in row] for row in values], seconds


def _info(client: _Client, path: str) -> dict:
    status, payload = client.send(path, None, path)
    if status != 200 or not isinstance(payload, dict):
        raise RuntimeError(f"{client.url}{path} did not return SGLang server information (HTTP {status}). "
                           f"Check --sglang-url, and set {API_KEY_ENV} for a server launched with --api-key.")
    return payload


def load_model(source: str, revision: str, *, url: str = DEFAULT_URL, timeout: float = DEFAULT_TIMEOUT_SECONDS,
               max_tokens: int = 4096):
    """Load the pinned reference tokenizer and verify one running SGLang server before any row."""
    local = Path(source).is_dir()
    if not (revision if local else re.fullmatch(r"[0-9a-f]{40}", revision or "")):
        raise ValueError("Remote sources require a pinned 40-character revision. "
                         "Local sources require a revision label.")
    url = url.rstrip("/")
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https") or "@" in parts.netloc or "?" in url or "#" in url:
        raise ValueError(f"--sglang-url must be an http or https URL without user info, query, or fragment, such as "
                         f"{DEFAULT_URL}. Set {API_KEY_ENV} for a server launched with --api-key.")
    launch = (f"python -m sglang.launch_server --model-path {source}" + ("" if local else f" --revision {revision}")
              + f" --port {parts.port or (443 if parts.scheme == 'https' else 80)}")
    client = _Client(url, timeout, f"Start one with {launch}, wait for its log line "
                                   "The server is fired up and ready to roll, or pass --sglang-url.")
    model_info, server_info = _info(client, "/model_info"), _info(client, "/server_info")
    served, limit = server_info.get("revision"), server_info.get("max_req_input_len")
    problem = None
    if model_info.get("model_path") != source:
        problem = (f"serves {model_info.get('model_path')}, but --model is {source}. Pass --sglang-url for a server "
                   f"that serves {source}, or launch one with --model-path {source}.")
    elif model_info.get("is_generation") is not True:
        problem = f"serves {source} as a non-generation model. Relaunch it without --is-embedding."
    elif model_info.get("load_format") == "dummy":
        problem = "was launched with --load-format dummy, which serves random weights. Relaunch without it."
    elif not local and served != revision:
        problem = f"serves revision {served}, but --revision is {revision}. Relaunch it with --revision {revision}."
    elif server_info.get("enable_mis"):
        problem = ("was launched with --enable-mis, which inserts delimiter tokens into the scored input. "
                   "Relaunch without it.")
    elif server_info.get("allow_auto_truncate"):
        problem = "was launched with --allow-auto-truncate, refused as a second truncation guard. Relaunch without it."
    elif model_info.get("preferred_sampling_params"):
        problem = (f"sets --preferred-sampling-params {json.dumps(model_info['preferred_sampling_params'])}, which "
                   "applies to every scoring request and can change the scores. Relaunch without it.")
    elif not isinstance(limit, int) or max_tokens >= limit:
        problem = (f"refuses inputs of {limit} tokens or more (max_req_input_len), so --max-tokens {max_tokens} "
                   f"is too large. Use a --max-tokens below {limit}, or serve the model with a larger context.")
    if problem:
        raise RuntimeError(f"The SGLang server at {url} {problem}")

    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        source, revision=None if local else revision, local_files_only=local, trust_remote_code=False)
    probe = {"id": "sglang-probe", "state": "probe evidence", "question": "probe criterion?",
             "options": [{"id": "yes", "description": "Yes."}, {"id": "no", "description": "No."}]}
    texts = [tokenizer.apply_chat_template(direct_messages(probe), tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False), *LETTERS, "yes", "no"]
    expected = [tokenizer.encode(text, add_special_tokens=False) for text in texts]
    status, payload = client.send("/v1/tokenize", {"prompt": texts, "add_special_tokens": False}, "the tokenizer check")
    if not isinstance(payload, dict) or payload.get("tokens") != expected:
        detail = "different token ids" if status == 200 else f"HTTP {status}: {_message(payload)}"
        raise RuntimeError(f"The SGLang server at {url} does not tokenize the probe prompt, the answer letters, yes, "
                           f"and no like {source}@{revision} ({detail}). Serve the same checkpoint and revision "
                           "with its own tokenizer.")
    subject = "the load-time score probe"
    try:
        _score(client, [expected[0]], [_slot_ids(tokenizer, len(LETTERS))], subject, [subject])
    except RuntimeError as error:
        raise RuntimeError(f"{str(error).rstrip('.')}. The server needs per-item candidate scoring with "
                           "return_token_logprobs (sgl-project/sglang#40826), in SGLang main from commit 174a5f37 "
                           "(2026-09-24) on or a nightly from 0.5.21.dev20260925 on.") from error
    info = {**server_info, **model_info}
    metadata = {"source": source, "revision": revision, "backend": "sglang", "server_url": url,
                "max_prompt_tokens": max_tokens, "transformers_version": transformers.__version__,
                "server": {key: info.get(key) for key in SERVER_FIELDS}}
    return client, tokenizer, metadata


def _result(row: dict, encoded, logprobs: list[float], metadata: dict, mode: str) -> dict:
    ids, slots, prompt_hash = encoded
    return {
        "id": row["id"],
        "option_ids": [option["id"] for option in row["options"]],
        "probabilities": softmax(logprobs),
        "option_logits": logprobs,
        "answer_token_ids": slots,
        "input_tokens": len(ids),
        "allowed_token_mass": sum(math.exp(value) for value in logprobs),
        "prompt_sha256": prompt_hash,
        "prompt_version": PROMPT_VERSION,
        "model": {**metadata, "serving_config": f"sglang-score-{mode}-v1"},
        "readout": READOUT,
        "probability_status": "conditional option score, uncalibrated as decision confidence",
    }


def _one_row(model, tokenizer, row: dict, metadata: dict, max_tokens: int, serial: bool) -> dict:
    started = time.perf_counter()
    encoded = encode_prompt(tokenizer, row, max_tokens)
    ids, extra = encoded[0], {}
    if serial:
        prefix = _state_prefix(tokenizer, row["state"])
        if not prefix or ids[: len(prefix)] != prefix or len(ids) <= len(prefix):
            raise ValueError(f"Row {row['id']}: state prefix does not match the full prompt. "
                             "--mode direct sends the same SGLang requests without this check")
        extra = {"prefix_tokens": len(prefix), "prefix_sha256": hashlib.sha256(json.dumps(prefix).encode()).hexdigest()}
    subject = f"row {row['id']}"
    (logprobs,), seconds = _score(model, [ids], [encoded[1]], subject, [subject])
    return {**_result(row, encoded, logprobs, metadata, "serial" if serial else "direct"), **extra,
            "request_seconds": seconds, "total_seconds": time.perf_counter() - started}


def score(model, tokenizer, row: dict, metadata: dict, max_tokens: int = 4096) -> dict:
    return _one_row(model, tokenizer, row, metadata, max_tokens, serial=False)


class SerialPrefixScorer:
    """Score rows in input order under the serial input rules, keyed on each row's own prefix tokens."""

    def __init__(self, model, tokenizer, metadata: dict, max_tokens: int = 4096):
        self.model, self.tokenizer, self.metadata, self.max_tokens = model, tokenizer, metadata, max_tokens

    def score(self, row: dict) -> dict:
        return _one_row(self.model, self.tokenizer, row, self.metadata, self.max_tokens, serial=True)


def score_shared(model, tokenizer, rows: list[dict], metadata: dict, max_tokens: int = 4096):
    """Send every row over one exact state as one /v1/score request, one item per row."""
    if not rows or any(row["state"] != rows[0]["state"] for row in rows[1:]):
        raise ValueError("Shared scoring requires one nonempty exact state")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Decision IDs must be unique")
    started = time.perf_counter()
    encoded = [encode_prompt(tokenizer, row, max_tokens) for row in rows]
    prefix = _state_prefix(tokenizer, rows[0]["state"])
    if not prefix or any(ids[: len(prefix)] != prefix or len(ids) <= len(prefix) for ids, _, _ in encoded):
        raise ValueError("The fixed state prefix does not match every full prompt. "
                         "--mode direct scores these rows with one SGLang request each")
    encode_seconds = time.perf_counter() - started
    scores, request_seconds = _score(model, [ids for ids, _, _ in encoded], [slots for _, slots, _ in encoded],
                                     f"the shared request of {len(rows)} rows", [f"row {row['id']}" for row in rows])
    results = [_result(row, row_encoded, logprobs, metadata, "shared")
               for row, row_encoded, logprobs in zip(rows, encoded, scores)]
    return results, {"total_seconds": time.perf_counter() - started, "encode_seconds": encode_seconds,
                     "request_seconds": request_seconds, "batch_size": len(rows), "prefix_tokens": len(prefix),
                     "true_suffix_tokens": sum(len(ids) - len(prefix) for ids, _, _ in encoded)}


def reranker_score(model, tokenizer, row: dict, metadata: dict, max_tokens: int = 4096) -> dict:
    """Score SemIf's reranker pairs of one row, one item per option with [no, yes] candidates."""
    validate_row(row)
    started = time.perf_counter()
    encoded = [reranker._encode(tokenizer, row, option, max_tokens) for option in row["options"]]
    no, yes = reranker._answer_ids(tokenizer)
    pairs, seconds = _score(model, [ids for ids, _ in encoded], [[no, yes]] * len(encoded), f"row {row['id']}",
                            [f"row {row['id']} option {option['id']}" for option in row["options"]])
    log_odds = [yes_value - no_value for no_value, yes_value in pairs]
    return {
        "id": row["id"],
        "option_ids": [option["id"] for option in row["options"]],
        "probabilities": softmax(log_odds),
        "option_logits": log_odds,
        "independent_binary_relevance": [softmax(pair)[1] for pair in pairs],
        "input_tokens": sum(len(ids) for ids, _ in encoded),
        "max_option_input_tokens": max(len(ids) for ids, _ in encoded),
        "option_prompt_sha256": [prompt_hash for _, prompt_hash in encoded],
        "prompt_version": reranker.PROMPT_VERSION,
        "model": {**metadata, "serving_config": "sglang-score-reranker-v1"},
        "readout": "SGLang /v1/score yes minus no log-odds per option, normalized only for relative comparison",
        "probability_status": "relative option compatibility, uncalibrated as categorical probability",
        "request_seconds": seconds,
        "total_seconds": time.perf_counter() - started,
    }
