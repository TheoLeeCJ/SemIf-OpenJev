# ROCm

SemIf's core Torch backend supports AMD GPUs through ROCm in direct, serial,
shared, and reranker modes. PyTorch intentionally exposes ROCm devices through
the `torch.cuda` API, so the SemIf device name remains `cuda`; there is no
`rocm` device value. The EXL3 bridge is separately pinned to CUDA 12.8 and is
not covered by this guide.

## Install

Install the ROCm build of the repository's exact Torch pin before installing
SemIf. The official ROCm 7.1 wheel index provides `torch==2.10.0` for supported
x86-64 Python versions:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/rocm7.1 \
  torch==2.10.0
python -m pip install -e '.[test]'
```

The resulting version has a local suffix such as `2.10.0+rocm7.1`; it satisfies
the existing `torch==2.10.0` project pin. Do not replace it with a CUDA or
CPU-only wheel. The host driver and GPU must also be supported by that ROCm
wheel.

## Isolate and validate one GPU

Select exactly one AMD GPU before starting Python or a scorer process:

```bash
export ROCR_VISIBLE_DEVICES=0
```

Before trusting the device, verify the wheel identity, actual `gfx` target,
compiled architecture list, and a synchronized BF16 matrix multiplication:

```bash
python - <<'PY'
import torch

assert torch.__version__.startswith("2.10.0+rocm7.1"), torch.__version__
assert torch.version.hip, torch.version.hip
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1, torch.cuda.device_count()

properties = torch.cuda.get_device_properties(0)
gfx = getattr(properties, "gcnArchName", "")
arches = torch.cuda.get_arch_list()
assert gfx.startswith("gfx"), gfx
target = gfx.split(":", 1)[0]
generic = f"{target[:5]}-generic"
assert target in arches or generic in arches, (target, generic, arches)
print("torch:", torch.__version__, "hip:", torch.version.hip)
print("device:", properties.name, "gfx:", gfx)
print("compiled architectures:", arches)

left = torch.randn((512, 512), device="cuda", dtype=torch.bfloat16)
right = torch.randn((512, 512), device="cuda", dtype=torch.bfloat16)
result = left @ right
torch.cuda.synchronize()
assert torch.isfinite(result).all().item()
print("BF16 matmul passed:", tuple(result.shape), result.dtype)
PY
```

Do not proceed with a workload if any assertion fails. In particular,
`torch.cuda.is_available()` alone cannot distinguish a working ROCm wheel from
an incorrectly installed runtime.

## Run

Use the same canonical `cuda` device used by PyTorch:

```bash
ROCR_VISIBLE_DEVICES=0 semif-score --mode direct --device cuda \
  --model Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --input examples/decisions.jsonl --output results-rocm-direct.jsonl
```

`--device auto` also selects the visible ROCm GPU. Every Torch result records
`torch_accelerator: "rocm"` and the HIP build version in its model metadata,
while `device` remains `cuda:0`. Timings and BF16 scores are hardware- and
kernel-sensitive; do not treat ROCm output as byte-identical to the committed
CUDA measurements.
