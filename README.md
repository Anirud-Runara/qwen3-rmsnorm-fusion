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
  fuse_model.py            # Offline in-place fusion + save

setup.py                   # Build the CUDA extension
```

> Quantization is intentionally out of scope for now — the quantization
> approach is still being finalised with the team.

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


