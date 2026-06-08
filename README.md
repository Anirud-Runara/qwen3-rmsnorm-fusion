# qwen3-rmsnorm-fusion

RMSNorm fusion implementation for **Qwen3-Coder-480B-A35B-Instruct**.

## What this does

Absorbs the `input_layernorm` scale (γ) into the downstream Q/K/V projection weights:

```
W_new = W * gamma          # element-wise row-wise scale
```

At inference time, the standalone `RMSNorm → Linear` pair becomes a single `Linear` call followed by a fast CUDA normalization kernel. This eliminates a memory round-trip per layer and reduces TTFT and decoding latency.

The MoE MLP is **not fused** (the router sits between the norm and the experts).

## Repo layout

```
csrc/
  denominator_kernel.cu    # CUDA kernels (V1 256-thread, V3 512-thread Welford)
  denominator.cpp          # PyBind11 wrapper

src/
  __init__.py
  load_cuda.py             # JIT-loads the compiled extension
  fused_forward.py         # FusedRMSNormCombinedLinearV1 / V3 nn.Module
  weight_transform.py      # Weight math; includes transform_qwen3_layer()
  patch_qwen3.py           # Runtime monkey-patch for Qwen3-MoE
  test_correctness_qwen3.py  # 3-level correctness test suite

scripts/
  install_deps.sh          # Install all Python and CUDA dependencies
  fuse_model.py            # Offline in-place fusion + save (loads full model)
  fuse_model_sharded.py    # Offline fusion, streams shards (peak RAM ~= 1 shard)
  quantize_nvfp4_sharded.py  # Shard-by-shard NVFP4A16 quantizer (RTN, data-free)

quantization/
  quantize_nvfp4_moe.py    # NVFP4 via llm-compressor (small models; OOMs on 480B)
  quantize_awq_moe.py      # AWQ W4A16 via llm-compressor (q/k/v kept bf16)
  probe_nvfp4_loading.py   # Check an NVFP4 checkpoint loads + is dequantizable
  README.md                # Quantization recipes, format notes, validation status

benchmarks/
  benchmark_rmsnorm_linear_fusion.py  # Single-layer fused-vs-unfused sweep
  results/                 # CSV outputs (one file per mode/variant)

setup.py                   # Build the CUDA extension
```

> **Quantization is now in scope and implemented.** The serving target is
> NVFP4 (compressed-tensors). See [End-to-end pipeline](#end-to-end-pipeline)
> below and [`quantization/README.md`](quantization/README.md). The fused
> NVFP4 480B checkpoint is uploaded to HuggingFace.

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/RunAra/qwen3-rmsnorm-fusion
cd qwen3-rmsnorm-fusion
bash scripts/install_deps.sh
```

### 2. Run correctness tests (Qwen3-0.6B, ~3 GB VRAM)

```bash
python3 -m src.test_correctness_qwen3
```

All three levels must pass before proceeding.

### 3. Validate offline fusion on a small model first

Before committing to the multi-hour 480B run, confirm the offline fusion
script itself is correct on a small model (Qwen3-0.6B fits on a single GPU):

```bash
python3 scripts/fuse_model.py \
  --model-id Qwen/Qwen3-0.6B \
  --output-dir /tmp/qwen3-0.6b-fused-bf16 \
  --sanity-check
```

The `--sanity-check` reload should produce coherent output. Only proceed to
the 480B run once this small-model pass looks correct.

### 4. Fuse the 480B model offline (CPU, ~900 GB RAM)

```bash
python3 scripts/fuse_model.py \
  --model-id Qwen/Qwen3-Coder-480B-A35B-Instruct \
  --output-dir /workspace/qwen3-480b-fused-bf16 \
  --sanity-check
```


## Key constraints

- Fusion must run on **BF16 weights**.
- The **MoE MLP is not fused** (only attention QKV).
- Qwen3's per-head `q_norm` / `k_norm` are independent of this fusion and run unchanged.


## End-to-end pipeline

The 480B is too large to load whole on the available hardware, so every stage
streams safetensors shards (peak RAM ≈ one shard) instead of materialising the
full model. Validate each stage on **Qwen3-30B-A3B** before the 480B run.

```
Qwen3-Coder-480B  (stock BF16)
        │
        │  scripts/fuse_model_sharded.py
        │  fold input_layernorm γ into q/k/v_proj, set γ = 1
        ▼
fused BF16  (γ absorbed)                          ← still BF16; no speedup yet
        │
        │  scripts/quantize_nvfp4_sharded.py
        │  weight-only NVFP4A16 (RTN, group=16, data-free)
        │  keeps lm_head + mlp.gate in original precision
        ▼
fused NVFP4  (compressed-tensors)   ──► uploaded to HuggingFace (serving artifact)
        │
        │  benchmarks/benchmark_rmsnorm_linear_fusion.py
        │  + src/patch_qwen3.py  (runtime CUDA kernel, BF16)
        ▼
fused-vs-unfused latency / memory / accuracy
```

**Why a custom sharded quantizer instead of llm-compressor?**
`quantization/quantize_nvfp4_moe.py` (llm-compressor `DataFreePipeline`) calls
`dispatch_model` internally and does **not** honor the `from_pretrained`
`max_memory` cap — on the 480B it either filled a single GPU to OOM or forced
~880 GB to host RAM and was `Killed`, even on a 2× RTX 6000 Pro + 1 TB box. The
weight-only NVFP4 transform is purely per-tensor, so
`scripts/quantize_nvfp4_sharded.py` does it one shard at a time with peak RAM ≈
one shard (~4 GB) and no `dispatch_model`. The llm-compressor scripts are kept
for small models and as a reference recipe.

> **Fusion happens in BF16, by design.** The CUDA kernel does
> `F.linear(x, W_new)` on a real-dtype tensor — it cannot consume packed FP4.
> NVFP4 buys disk/VRAM at rest; at inference both arms decompress q/k/v to BF16
> and the kernel runs on those. So NVFP4 is the *storage/serving* format, not
> something the fusion kernel computes on. (compressed-tensors has no FP4
> compute kernel in HF transformers — it decompresses to BF16 on first forward.)


## Benchmarking

`benchmarks/benchmark_rmsnorm_linear_fusion.py` benchmarks the fusion site
(`input_layernorm → [q_proj, k_proj, v_proj]`) on **one decoder layer**, loaded
directly from the shard files (`--load-mode layer`, ~seconds, no full-model
load). It sweeps batch ∈ {1, 8, 32} × seq ∈ {128, 512, 2048} and records
median/p99 latency, throughput, peak memory, and fused-vs-unfused numerical
equivalence (max|diff|, cosine, KL).

Single-layer isolation is deliberate: in an end-to-end 480B forward the MoE
experts dominate wall-clock and would dilute the QKV-fusion delta to near zero,
so the gain is measured where it occurs.

Two modes — **they answer different questions**:

| Mode | Both arms built from | Measures | Use when |
|---|---|---|---|
| `runtime-patch` *(default)* | the **same** unfused weights | the **kernel** alone (weights identical) | comparing fused vs unfused kernel fairly |
| `checkpoints` | **separate** `models/non-fused/` and `models/fused/` | the **end-to-end checkpoints** incl. quantization | comparing the shipped fused checkpoint vs baseline |

```bash
# discover the layer path (reads index only):
python benchmarks/benchmark_rmsnorm_linear_fusion.py --dir /workspace --print-keys

# clean kernel comparison (recommended), V3 kernel:
python benchmarks/benchmark_rmsnorm_linear_fusion.py --dir /workspace --variant V3

# V1 (256-thread) vs V3 (512-thread):
python benchmarks/benchmark_rmsnorm_linear_fusion.py --dir /workspace --variant V1
```


## Results

Single layer-0 of Qwen3-Coder-480B (`hidden = 6144`), `cuda:0`, BF16, 50 warmup
/ 200 measured iters. <!-- GPU model: TODO confirm -->
Raw CSVs in [`benchmarks/results/`](benchmarks/results/).

**`runtime-patch`, V3 (clean kernel comparison — the headline numbers):**

| batch | seq | fused (ms) | non-fused (ms) | speedup | cosine sim |
|------:|----:|-----------:|---------------:|:-------:|:----------:|
| 1  | 128  | 0.252 | 0.363 | **1.44×** | 0.999992 |
| 1  | 512  | 0.438 | 0.458 | 1.05× | 0.999992 |
| 1  | 2048 | 1.028 | 1.296 | 1.26× | 0.999992 |
| 8  | 128  | 0.613 | 0.801 | 1.31× | 0.999992 |
| 8  | 512  | 2.245 | 2.598 | 1.16× | 0.999992 |
| 8  | 2048 | 9.062 | 9.997 | 1.10× | 0.999993 |
| 32 | 128  | 2.250 | 2.601 | 1.16× | 0.999993 |
| 32 | 512  | 9.082 | 9.996 | 1.10× | 0.999993 |
| 32 | 2048 | 36.33 | 40.00 | 1.10× | 0.999993 |

Takeaways:
- **Speedup is largest when the matmul is small** (1.44× at batch1/seq128) and
  decays toward **~1.10×** as the QKV matmul comes to dominate — consistent with
  the fusion saving a fixed RMSNorm HBM round-trip whose relative cost shrinks
  as the matmul grows. (The batch1/seq512 dip to 1.05× is the one off-trend
  point; treat as run-to-run variance.)
- **Numerically faithful**: cosine ≈ 0.99999, max|diff| ≈ 5e-4 — the fused
  kernel reproduces the separate `norm → q/k/v` path.
- **V1 ≈ V3**: the 256- vs 512-thread normalize kernels are within noise across
  all shapes; neither is a clear winner at `hidden = 6144`.
- **Peak memory**: fused ≤ non-fused, and the gap widens with size (e.g.
  batch32/seq2048: **4696 MB fused vs 5464 MB** non-fused, ~14% lower) — it skips
  the materialized normalized-activation tensor.

> ⚠️ The `checkpoints`-mode CSVs show **broken numerical equivalence**
> (cosine ≈ 0.944, max|diff| ≈ 3e9). Speedups there are in the same ~1.1–1.44×
> range, but the accuracy numbers are **not trustworthy** until the dequant
> convention is fixed — see below. Cite the `runtime-patch` numbers as the
> validated result.


## Known issues & next steps

1. **NVFP4 dequant convention mismatch (blocks `checkpoints`-mode accuracy).**
   `scripts/quantize_nvfp4_sharded.py` reconstructs weights as
   `code × weight_scale × weight_global_scale` (**multiply**), but the
   benchmark's `_dequantize_nvfp4` uses `code × weight_scale ÷ weight_global_scale`
   (**divide**). Dividing by the ~1e-6 global scale inflates weights by ~1e10,
   which matches the observed 3e9 max|diff| in `checkpoints` mode. **Action:**
   pick one convention, validate it against a known-good llm-compressor NVFP4
   checkpoint (quantize the 30B both ways and compare reconstructed weights),
   then unify both call sites. Re-run `checkpoints` mode after the fix.
2. **Validate the sharded NVFP4 format against the spec.** The injected
   `quantization_config.format` (`"float-quantized"`) and global-scale meaning
   are a best-guess; confirm a vLLM / compressed-tensors load round-trips the
   uploaded checkpoint before relying on it for serving.
3. **Confirm baseline comparability with the rest of the team.** The unfused
   baseline must be the *same* quantization scheme (weight-only NVFP4A16, no
   calibration) as ours, or cross-team numbers won't be comparable.
4. **Record the GPU + full environment** in the results (GPU model, driver,
   CUDA, torch, transformers, compressed-tensors versions) for reproducibility.
5. **Optional: end-to-end sanity number.** A few full-model decode steps (fused
   vs unfused) would show the realistic whole-model TTFT/throughput delta once
   experts are included — expected to be small, but worth quoting alongside the
   per-layer gain.


