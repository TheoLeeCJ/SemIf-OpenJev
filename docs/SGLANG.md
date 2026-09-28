# SGLang server backend

`--backend sglang` runs SemIf's direct, serial, shared, and reranker modes through a running [SGLang](https://github.com/sgl-project/sglang) server. SemIf still renders every prompt, checks the answer slots, and validates the input on the pinned reference tokenizer, so `prompt_sha256`, `input_tokens`, `prompt_version`, option order, and `answer_token_ids` are the values the Torch backend computes for the same row. SGLang only scores SemIf's own token ids through `POST /v1/score` with the per-item candidates and `return_token_logprobs` that mickqian added in [sgl-project/sglang#40826](https://github.com/sgl-project/sglang/pull/40826). The scorer adds no Python dependency and runs no model.

## Requirements

- An SGLang build that contains sgl-project/sglang#40826, which is SGLang main from commit 174a5f37 (2026-09-24) on or a nightly wheel from 0.5.21.dev20260925 on. Release 0.5.20 and earlier lack it, and the backend refuses such a server.
- One GPU per served model. The server owns the device and the precision, so `--device` and `--dtype` do not apply, as with MLX and llama.cpp.

SGLang's [install guide](https://github.com/sgl-project/sglang/blob/acc15c193b9d8813cb31b66cd19980fe0de9d72d/docs/docs/get-started/install.mdx) describes nightly and source installs. Its nightly command is:

```bash
pip install --upgrade pip
pip install uv
uv pip install --prerelease=allow --index-strategy unsafe-best-match --extra-index-url https://docs.sglang.ai/whl/cu130/ sglang
```

## Launch a server

Run each server from the SGLang environment in its own terminal. The direct model listens on SGLang's default port, which is also the default `--sglang-url`, and the reranker gets the next port and the next GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server --model-path Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --host 127.0.0.1 --port 30000
CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server --model-path Qwen/Qwen3-Reranker-4B \
  --revision 22e683669bc0f0bd69640a1354a6d0aebcfeede5 --host 127.0.0.1 --port 30001
```

`--revision` is required for a Hub model, because the backend refuses a server that reports no revision or another one. Wait for the server to log `The server is fired up and ready to roll!` before scoring.

## Score

From the repository root, in SemIf's own environment:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'

semif-score --backend sglang --mode direct \
  --model Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --input examples/decisions.jsonl \
  --output results-sglang-direct.jsonl

semif-score --backend sglang --mode reranker --sglang-url http://127.0.0.1:30001 \
  --model Qwen/Qwen3-Reranker-4B \
  --revision 22e683669bc0f0bd69640a1354a6d0aebcfeede5 \
  --input examples/decisions.jsonl \
  --output results-sglang-reranker.jsonl
```

`--model` and `--revision` must name exactly what the server serves, and `--mode serial` and `--mode shared` use the direct model's server. For a server launched with `--api-key`, set `SEMIF_SGLANG_API_KEY` to that key, which SemIf sends only as a bearer header and never writes to a row. Requests go straight to the server without proxy variables or redirects, and `--sglang-timeout` (default 300 seconds) bounds the wait for each answer.

## What each mode sends

- direct and serial: one request per row, with the row's token ids as one item and its answer slots as that item's candidates. Serial adds the serial input rules and the client-side `prefix_tokens` and `prefix_sha256`. The server's radix cache can reuse a prefix from earlier requests, which the response does not report, so launch with `--disable-radix-cache` for values computed without prefix reuse.
- shared: one request for the whole input, with one item per row, so the timeout covers every row. The server admits the items as independent requests, so shared mode on SGLang does not guarantee that the state is prefilled once.
- reranker: one request per row, with one item per option and `[no, yes]` as the candidates of every item, built from SemIf's reranker prompt.

## Rows

- `option_logits` holds the candidates' full-vocabulary log-probabilities, which differ from raw logits by one constant per row, as the `readout` string says. Softmax, argmax, and temperature calibration are unchanged by that constant, but the values are not comparable with Torch `option_logits`. Reranker `option_logits` stay the yes minus no log-odds.
- `probabilities` is SemIf's own softmax, and as on every backend these are conditional option scores, not calibrated decision confidence. They come from SGLang's kernels and scheduler, so they can differ from the Torch backend, and some rows differ by more than 0.05, the review threshold of `manifests/mlx-validation.json`. Compare decisions or probabilities with a tolerance.
- `request_seconds`, `total_seconds`, and `shared_timing` are client wall time including HTTP and server queueing. `cache_hit`, `forward_seconds`, `prefill_seconds`, and `full_vocab_argmax_id` are omitted because `/v1/score` does not report them.
- `model` records an allowlist of server facts from `/model_info` and `/server_info`, never the server's API keys or launch command.

## Refused servers

Before the first row, the backend reads `/model_info` and `/server_info`, compares the server tokenizer with the reference tokenizer through `/v1/tokenize`, and sends one small `/v1/score` probe. Each refusal names the fix. It refuses a server it cannot reach (including one still loading), a server without sgl-project/sglang#40826, another model or revision, an embedding model, `--load-format dummy` (random weights), `--enable-mis` (delimiter tokens in the scored input), `--allow-auto-truncate` (a second guard, since every scored input stays below the server input limit), any `--preferred-sampling-params` (applied to every scoring request), a server tokenizer that disagrees on the probe prompt, an answer letter, `yes`, or `no`, and a `--max-tokens` at or above the server's `max_req_input_len` (the server refuses an input of that many tokens).

Every response must carry one finite log-probability per candidate and a `usage.prompt_tokens` equal to the token ids sent. A failed request or check stops the run with the row id and the server's message, and nothing is retried. The checks run once when the scorer starts, so a server restart during a run is not detected.

## Tests

`pytest -q` runs the offline backend tests against a fake server. With the two servers above running, this also runs the real-server tests:

```bash
SEMIF_SGLANG_URL=http://127.0.0.1:30000 SEMIF_SGLANG_RERANKER_URL=http://127.0.0.1:30001 pytest -q tests/test_sglang.py
```
