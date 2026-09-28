"""Create-only JSONL command line scorer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import load_causal_model, validate_row
from .direct import score as direct_score
from .reranker import score as reranker_score
from .serial import SerialPrefixScorer
from .shared import score_shared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("direct", "serial", "shared", "reranker"), required=True)
    parser.add_argument("--backend", choices=("torch", "mlx", "llamacpp", "sglang"), default="torch")
    parser.add_argument("--mlx-bits", type=int, choices=(4, 8), help="Quantize MLX weights in memory; default preserves source precision")
    parser.add_argument("--mlx-cache-limit-mib", type=int,
                        help="MLX inactive allocation cache in MiB (default: 256; 0 disables caching)")
    parser.add_argument("--gguf", type=Path, help="Local GGUF checkpoint for --backend llamacpp")
    parser.add_argument("--llama-threads", type=int,
                        help="CPU threads for --backend llamacpp (default: all visible cores)")
    parser.add_argument("--sglang-url", help="SGLang server URL for --backend sglang, which owns device and precision "
                        "(default: http://127.0.0.1:30000)")
    parser.add_argument("--sglang-timeout", type=float, help="Seconds to wait for each SGLang request. Shared mode "
                        "sends every row in one request (default: 300)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto",
                        help="Torch device (auto prefers CUDA, then Apple MPS; CPU must be explicit)")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16",
                        help="Model precision; changing it can change option scores")
    args = parser.parse_args()
    if args.output.exists() or args.max_tokens < 1:
        parser.error("Output must be new and max-tokens must be positive")
    if args.mlx_bits and args.backend != "mlx":
        parser.error("--mlx-bits requires --backend mlx")
    if args.mlx_cache_limit_mib is not None:
        if args.backend != "mlx":
            parser.error("--mlx-cache-limit-mib requires --backend mlx")
        if args.mlx_cache_limit_mib < 0:
            parser.error("--mlx-cache-limit-mib must be nonnegative")
    if args.gguf is not None and args.backend != "llamacpp":
        parser.error("--gguf requires --backend llamacpp")
    if args.llama_threads is not None:
        if args.backend != "llamacpp":
            parser.error("--llama-threads requires --backend llamacpp")
        if args.llama_threads < 1:
            parser.error("--llama-threads must be positive")
    if args.backend == "mlx" and args.mode == "reranker":
        parser.error("MLX supports direct, serial, and shared modes; reranker requires torch")
    if args.backend == "llamacpp":
        if args.mode == "reranker":
            parser.error("llama.cpp supports direct, serial, and shared modes; reranker requires torch")
        if args.gguf is None or not args.gguf.is_file():
            parser.error("--backend llamacpp requires --gguf pointing at an existing GGUF file")
    if args.backend != "sglang" and (args.sglang_url is not None or args.sglang_timeout is not None):
        parser.error("--sglang-url and --sglang-timeout require --backend sglang")
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if not rows:
        parser.error("Input is empty")
    for row in rows:
        validate_row(row)
    direct, serial, shared, reranker = direct_score, SerialPrefixScorer, score_shared, reranker_score
    if args.backend == "mlx":
        from . import mlx_backend

        cache_limit_mib = (mlx_backend.DEFAULT_CACHE_LIMIT_MIB if args.mlx_cache_limit_mib is None
                           else args.mlx_cache_limit_mib)
        model, tokenizer, metadata = mlx_backend.load_model(
            args.model, args.revision, args.mlx_bits, cache_limit_mib=cache_limit_mib)
        direct, serial, shared = mlx_backend.score, mlx_backend.SerialPrefixScorer, mlx_backend.score_shared
    elif args.backend == "llamacpp":
        from . import llamacpp_backend

        model, tokenizer, metadata = llamacpp_backend.load_model(
            args.model, args.revision, args.gguf,
            threads=args.llama_threads, context_tokens=args.max_tokens)
        direct, serial, shared = (llamacpp_backend.score, llamacpp_backend.SerialPrefixScorer,
                                  llamacpp_backend.score_shared)
    elif args.backend == "sglang":
        from . import sglang_backend

        model, tokenizer, metadata = sglang_backend.load_model(
            args.model, args.revision, max_tokens=args.max_tokens,
            url=sglang_backend.DEFAULT_URL if args.sglang_url is None else args.sglang_url,
            timeout=sglang_backend.DEFAULT_TIMEOUT_SECONDS if args.sglang_timeout is None else args.sglang_timeout)
        direct, serial, shared, reranker = (sglang_backend.score, sglang_backend.SerialPrefixScorer,
                                            sglang_backend.score_shared, sglang_backend.reranker_score)
    else:
        if args.mode == "reranker":
            if args.device in {"mps", "cpu"}:
                parser.error(f"Reranker mode requires CUDA; --device {args.device} is unsupported")
            # Keep auto CUDA-only and let the loader enforce one visible GPU.
            args.device = "cuda"
        model, tokenizer, metadata = load_causal_model(args.model, args.revision, args.device, args.dtype)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as destination:
        if args.mode == "shared":
            results, timing = shared(model, tokenizer, rows, metadata, args.max_tokens)
            for result in results:
                destination.write(json.dumps({**result, "shared_timing": timing}, allow_nan=False) + "\n")
        elif args.mode == "serial":
            scorer = serial(model, tokenizer, metadata, args.max_tokens)
            for row in rows:
                destination.write(json.dumps(scorer.score(row), allow_nan=False) + "\n")
                destination.flush()
        else:
            scorer = direct if args.mode == "direct" else reranker
            for row in rows:
                destination.write(json.dumps(scorer(model, tokenizer, row, metadata, args.max_tokens), allow_nan=False) + "\n")
                destination.flush()


if __name__ == "__main__":
    main()
