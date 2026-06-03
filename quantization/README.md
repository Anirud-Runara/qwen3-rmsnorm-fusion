# Quantization (AWQ W4A16, fusion-compatible)

AWQ 4-bit quantization of the **fused** Qwen3-Coder-480B using
[llm-compressor](https://github.com/vllm-project/llm-compressor), built to
coexist with the RMSNorm-fusion V1 kernel.

## Reference recipe (QuantTrio)

[`QuantTrio/Qwen3-Coder-480B-A35B-Instruct-AWQ`](https://huggingface.co/QuantTrio/Qwen3-Coder-480B-A35B-Instruct-AWQ)
(the checkpoint used in earlier benchmarking) uses, per its `config.json`:

```json
{ "quant_method": "awq", "bits": 4, "group_size": 128,
  "version": "gemm", "zero_point": true, "modules_to_not_convert": [".mlp.gate"] }
```

i.e. **4-bit, group 128, asymmetric, everything quantized except the MoE router
gate**. QuantTrio does **not** publish their script or calibration dataset (their
"how did you do it?" discussion is unanswered), so we rebuild the recipe with
llm-compressor and our own calibration data.

## How ours differs — and why

QuantTrio quantizes attention **q/k/v** too. Our V1 fusion kernel does a **bf16**
matmul on the combined QKV and cannot consume INT4 q/k/v, and AWQ's scale
absorption would disturb the `input_layernorm` (gamma=1) we fused. So this script
**keeps q/k/v in bf16** (adds them to the ignore list) and quantizes everything
else (experts, o_proj, ...). q/k/v are a tiny fraction of the 480B, so we still
capture nearly all the memory savings. `.mlp.gate` stays bf16, like QuantTrio.

> Benchmark-design note: this makes our model *not identical* to QuantTrio's
> (our QKV is bf16). Reconcile this with the team before comparing numbers.

## ⚠️ Environment: use a dedicated venv

llm-compressor likely **does not support transformers 5.x yet** (the serving
box runs 5.9). Quantize in a separate venv and let pip choose the transformers
version — do not pin 5.9:

```bash
python3 -m venv .venv-quant && source .venv-quant/bin/activate
pip install llmcompressor datasets
python3 -c "import transformers, llmcompressor; print(transformers.__version__, llmcompressor.__version__)"
```

The output checkpoint is just files — you serve it later in your transformers-5.x
env with `patch_qwen3_model(variant="V1")`. Quantize and serve in different envs.

## Validate on the small MoE first

```bash
python3 quantize_awq_moe.py --model-id Qwen/Qwen3-30B-A3B --output-dir ./qwen3-30b-a3b-awq
```
Confirm, on the 30B:
1. llm-compressor runs on Qwen3-MoE **without NaNs** (MoE experts calibrate — see below).
2. The ignore list leaves q/k/v in bf16 (check the saved `config.json` /
   `recipe.yaml`).
3. Quantized model **+ `patch_qwen3_model(variant="V1")`** still gives correct
   logits (reuse the integration test). For the full fusion+quant path, fuse the
   30B first (`scripts/fuse_model_sharded.py`) and quantize that.
4. Size / quality look sane.

## Then the fused 480B

```bash
python3 quantize_awq_moe.py \
  --model-id /workspace/qwen3-480b-fused-bf16 \
  --output-dir /workspace/qwen3-480b-fused-awq \
  --device-map auto
```
480B needs offload/multi-GPU — llm-compressor's sequential pipeline processes one
layer at a time, so a high-RAM box with 1–few GPUs can work (it does not need the
whole model in VRAM at once).

## Open items to verify on the 30B run

- **MoE calibration:** llm-compressor 0.9 added an "updated MoE calibration
  context" that routes all tokens through all experts (without it, rarely-used
  experts under-calibrate -> NaNs). Confirm Qwen3-MoE is handled automatically;
  if you see NaNs, check `llmcompressor.modeling` for a Qwen3-MoE calibration
  module that must be applied explicitly (the GLM example uses
  `CalibrationGlm4MoeMoE`).
- **`AWQModifier` import path** varies by version — the script tries the known
  locations.
- **Calibration data:** defaults to `ise-uiuc/Magicoder-Evol-Instruct-110K`
  (code-instruction), chat-template formatted. Swap via `--calib-dataset`;
  match the inference workload (agentic coding). 512 samples is a starting point.
