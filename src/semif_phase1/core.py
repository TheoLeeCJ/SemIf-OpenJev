"""Shared input validation, prompts, model loading, and numeric helpers."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

LETTERS = "ABCDEFGHIJKLMNOP"
DIRECT_SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)


def validate_row(row: dict) -> None:
    required = {"id", "state", "question", "options"}
    if not required <= row.keys():
        raise ValueError(f"Row is missing fields: {sorted(required - row.keys())}")
    if not all(isinstance(row[key], str) and row[key] for key in ("id", "question")):
        raise ValueError("id and question must be nonempty strings")
    state = row["state"]
    if not isinstance(state, (str, dict, list)) or not state:
        raise ValueError("state must be a nonempty string, object, or array")
    try:
        json.dumps(state, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("state must be finite JSON-compatible data") from error
    options = row["options"]
    if not isinstance(options, list) or not 2 <= len(options) <= len(LETTERS):
        raise ValueError("options must contain 2-16 entries")
    ids = []
    for option in options:
        if not isinstance(option, dict) or not isinstance(option.get("id"), str) or not isinstance(option.get("description"), str):
            raise ValueError("Each option needs string id and description fields")
        ids.append(option["id"])
    if len(ids) != len(set(ids)):
        raise ValueError("Option IDs must be unique")


def direct_messages(row: dict) -> list[dict]:
    validate_row(row)
    payload = {
        "evidence": row["state"],
        "criterion": row["question"],
        "options": [
            {"letter": LETTERS[index], "description": option["description"]}
            for index, option in enumerate(row["options"])
        ],
    }
    return [
        {"role": "system", "content": DIRECT_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def softmax(values: list[float]) -> list[float]:
    if len(values) < 2 or any(not math.isfinite(value) for value in values):
        raise ValueError("Need at least two finite scores")
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    total = sum(weights)
    return [weight / total for weight in weights]


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def torch_accelerator_metadata(torch, device) -> dict:
    """Identify the accelerator behind a resolved Torch device."""
    accelerator = getattr(device, "type", str(device).split(":", 1)[0])
    version = None
    if accelerator == "cuda":
        hip = getattr(getattr(torch, "version", None), "hip", None)
        if hip:
            accelerator, version = "rocm", hip
        else:
            version = getattr(getattr(torch, "version", None), "cuda", None)
    metadata = {"torch_accelerator": accelerator}
    if version:
        metadata["torch_accelerator_version"] = str(version)
    return metadata


def resolve_device(device: str = "auto"):
    import torch

    if device not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError("Device must be auto, cuda, mps, or cpu")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps"
    if device == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            if getattr(getattr(torch, "version", None), "hip", None):
                raise ValueError(
                    "Expose exactly one ROCm GPU through Torch's cuda device namespace, "
                    "for example with ROCR_VISIBLE_DEVICES=0"
                )
            raise ValueError("Expose exactly one CUDA GPU, for example with CUDA_VISIBLE_DEVICES=0")
        return torch.device("cuda:0")
    if device == "cpu":
        return torch.device("cpu")
    if not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable; use an Apple Silicon Mac with an MPS-enabled PyTorch build")
    return torch.device("mps")


def synchronize(device) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def load_causal_model(source: str, revision: str, device: str = "auto", dtype: str = "bfloat16"):
    """Load one pinned causal model on the resolved Torch device."""
    import torch
    import transformers

    local = Path(source).exists()
    if not local and not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
        raise ValueError("Remote models require a pinned 40-character commit revision")
    if local and not revision:
        raise ValueError("Local models require an explicit manifest/revision string")
    target = resolve_device(device)
    if dtype not in {"bfloat16", "float16", "float32"}:
        raise ValueError("Dtype must be bfloat16, float16, or float32")
    common = {"revision": None if local else revision, "local_files_only": local, "trust_remote_code": False}
    config = transformers.AutoConfig.from_pretrained(source, **common)
    tokenizer = transformers.AutoTokenizer.from_pretrained(source, **common)
    cls = transformers.AutoModelForCausalLM
    if config.model_type in {"qwen3_5", "qwen3_5_text"}:
        cls = getattr(transformers, "Qwen3_5ForCausalLM", None)
        if cls is None:
            raise RuntimeError("Installed transformers lacks the native Qwen3.5 model")
        config = config.get_text_config()
    model, loading = cls.from_pretrained(
        source,
        config=config,
        dtype=getattr(torch, dtype),
        device_map={"": str(target)},
        low_cpu_mem_usage=True,
        output_loading_info=True,
        **common,
    )
    if any(loading.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"Checkpoint did not load completely: {loading}")
    model.eval()
    metadata = {
        "source": source,
        "revision": revision,
        "dtype": dtype,
        "device": str(target),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        **torch_accelerator_metadata(torch, target),
    }
    return model, tokenizer, metadata
