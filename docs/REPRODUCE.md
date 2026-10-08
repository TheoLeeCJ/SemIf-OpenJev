# Reproduction guide

## Environment

Create an isolated virtual environment and place caches on a drive with room for model weights:

```bash
python -m venv .venv
. .venv/bin/activate
export HF_HOME=/path/to/large-drive/huggingface
pip install -r requirements.txt
pip install -e .
pytest -q
```

Use one GPU per scorer process. The measured environment was Ubuntu 22.04 on Linux x86_64, Python 3.10.12, NVIDIA driver 595.71.05, CUDA 12.8, PyTorch 2.10.0+cu128, Transformers 5.17.0, BF16, and an RTX 3090. `requirements.txt` pins the observed Python runtime packages; the CUDA-enabled PyTorch wheel still requires a compatible NVIDIA driver. Exact model commit IDs are in [../manifests/models.json](../manifests/models.json).

For AMD GPUs, follow the [ROCm installation and validation guide](ROCM.md).
ROCm is an additive Torch runtime path and does not change the NVIDIA environment
or the committed RTX 3090 measurements above.

`pytest -q` runs all core and browser-source tests. Timing is hardware-sensitive, and BF16/kernel differences can change borderline probabilities or choices. Treat committed row counts, schemas, source hashes, and checksums as exact acceptance criteria; treat timings and model outputs as measurements to compare with the committed row-level evidence, not byte-identical golden outputs.

## Score owned examples

```bash
CUDA_VISIBLE_DEVICES=0 semif-score --mode direct \
  --model Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --input examples/decisions.jsonl --output results-direct.jsonl

CUDA_VISIBLE_DEVICES=0 semif-score --mode serial \
  --model Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --input examples/decisions.jsonl --output results-serial.jsonl

CUDA_VISIBLE_DEVICES=0 semif-score --mode reranker \
  --model Qwen/Qwen3-Reranker-4B \
  --revision 22e683669bc0f0bd69640a1354a6d0aebcfeede5 \
  --input examples/decisions.jsonl --output results-reranker.jsonl
```

The command refuses an existing output path and refuses silent input truncation. Each output embeds the exact revision, library versions, prompt hash, token count, timings, and an explicit probability-status warning. State may be a nonempty string, JSON object, or JSON array. `serial` caches consecutive equal states. `shared` requires every input row to carry the same exact state and is exercised by the 37×21 runner below.

## Third-party evaluations

TypeSafe source records are not included. To reproduce that comparison, supply local snapshots in the source directory. The helper fetches the remaining public evaluation inputs with hash verification:

```bash
python benchmarks/fetch_sources.py --output /path/on/large-drive/semif-sources
```

The frozen 706-row matrix and source IDs are in `benchmarks/manifests/`. Row-level direct and reranker outputs are in `results/raw/predictions/`. The complete owned 144-row labeled workload is distributed in `benchmarks/data/authored144.jsonl`.

Build the exact external evaluation rows and recompute their metrics with the commands in [the benchmark guide](../benchmarks/README.md#quality-evidence). The builders verify source hashes and frozen selection IDs; the TypeSafe and Every evaluators accept the rebuilt gold rows plus the committed row-level predictions.

## Reproduce perturbation evidence

Rebuild the frozen 108-row fixture from the 36 owned originals, then verify it matches the committed fixture:

```bash
python benchmarks/build_perturbations.py \
  --source benchmarks/data/authored144.jsonl \
  --output /tmp/perturbations108.jsonl \
  --manifest /tmp/perturbations108-manifest.json
cmp /tmp/perturbations108.jsonl benchmarks/data/perturbations108.jsonl
```

Regenerate direct and reranker predictions with `semif-score --mode serial` and `--mode reranker`, respectively, or recompute the exact committed report from the included row-level predictions:

```bash
python benchmarks/evaluate_perturbations.py \
  --gold benchmarks/data/authored144.jsonl \
  --perturbations benchmarks/data/perturbations108.jsonl \
  --direct-base results/raw/predictions/direct-authored144.jsonl \
  --direct-perturbations results/raw/predictions/direct-perturbations108.jsonl \
  --reranker-base results/raw/predictions/reranker-authored144.jsonl \
  --reranker-perturbations results/raw/predictions/reranker-perturbations108.jsonl \
  --output perturbation-report.json
cmp perturbation-report.json results/raw/perturbation-comparison.json
```

## Reproduce the headline speed results

Run the focused three-repeat direct-versus-compact-array comparison:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/decision_vs_generation.py \
  --model Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --input benchmarks/data/shape777.jsonl \
  --output compact-array-run.json
```

Run the complete 777-decision fresh, serial-cache, and parallel shared-state comparison:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/shape777.py \
  --model Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --input benchmarks/data/shape777.jsonl \
  --output shape777-run.json
```

Run the complete native-reranker comparison at the published pair batch sizes:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/shape777_reranker.py \
  --model Qwen/Qwen3-Reranker-4B \
  --revision 22e683669bc0f0bd69640a1354a6d0aebcfeede5 \
  --input benchmarks/data/shape777.jsonl \
  --pair-batch-sizes 1,4,8 \
  --output shape777-reranker-run.json
```

All scripts require a new output path. Timing includes prompt construction, tokenization, transfers, model execution, and CPU readout after a warmup; model loading and final result-file writes are excluded.

Verify the committed evidence bundle and confirm that every selected scalar in the machine-readable summary matches its raw report:

```bash
(cd results/raw && sha256sum -c SHA256SUMS)
python benchmarks/verify_published.py
```

The source-specific quality commands above regenerate the metrics stored in `results/raw/quality-comparison.json`. `verify_published.py` checks 69 published summary values against that report plus the perturbation, systems, and generation reports. It deliberately does not require byte-identical GPU reruns.

## exl3 bridge probe (quantized readout, additive track)

Row-level probe evidence and reproduction for the quantized-readout bridge
live in `exl3-bridge/` (see its README for the exact container invocation,
pinned exllamav3 runtime, and quantized checkpoint revision). Verify its
bundle with:

```bash
(cd exl3-bridge/results && sha256sum -c SHA256SUMS)
python -m pytest exl3-bridge/test_bridge.py -q
```

The bridge does not participate in the headline matrix and none of
`results/phase1-summary.json` applies to it.
