#!/usr/bin/env python3
"""
NVFP4 quantization for Qwen3-MoE with llm-compressor → compressed-tensors format.

WHY compressed-tensors (not modelopt):
  NVIDIA's published NVFP4 checkpoints are `modelopt` format, which plain HF
  transformers cannot load ("Unknown quantization type: modelopt"). We benchmark
  in HF + a monkey-patched CUDA kernel, so we need an *HF-native* NVFP4 format.
  llm-compressor emits `compressed-tensors`, which transformers loads directly
  and which exposes `CompressedLinear.compressor.decompress_module()` — the exact
  dequant hook the fusion patch uses to get a dense BF16 weight for the kernel.

WHAT TO QUANTIZE (two valid setups — point --model-id at either):
  - Unfused stock Qwen3-Coder-480B  → ONE NVFP4 checkpoint; baseline = stock,
    fused arm = patch it at runtime (cleanest comparison, single source).
  - Your already-fused BF16 checkpoint → NVFP4; use as the fused arm directly
    (γ already absorbed). Note: two-pipeline confound vs the nvidia baseline.

MEMORY (answers "do I load all weights?"):
  No — oneshot runs a SEQUENTIAL pipeline (one layer on GPU at a time, rest
  offloaded to CPU/disk). Peak GPU ≈ one layer + activations. But the full
  checkpoint must be on disk + enough CPU RAM (or --offload-dir) to stream it.
  Validate on Qwen3-30B-A3B (one GPU) before the 480B.

CALIBRATION (answers "do I need a dataset?"):
  Yes. Default scheme NVFP4 = W4A4 (weights AND activations 4-bit) → activation
  scales need a calibration pass. MoE experts must see data (routed through all
  experts) or they under-calibrate → NaNs. Use code-instruction data for this
  coding model. Use --scheme NVFP4A16 for weight-only (closer to data-free, but
  not the production scheme).

USAGE
  # validate on the small MoE first (one GPU):
  python3 quantization/quantize_nvfp4_moe.py \
      --model-id Qwen/Qwen3-30B-A3B --output-dir ./qwen3-30b-nvfp4

  # then the real run (point at the 480B; high-RAM box or --offload-dir):
  python3 quantization/quantize_nvfp4_moe.py \
      --model-id /workspace/qwen3-480b-bf16 \
      --output-dir /workspace/qwen3-480b-nvfp4 \
      --offload-dir /workspace/offload
"""

import argparse

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier


def parse_args():
    p = argparse.ArgumentParser(description="NVFP4 (compressed-tensors) quantization for Qwen3-MoE")
    p.add_argument("--model-id", required=True,
                   help="HF id or local path of the BF16 checkpoint to quantize")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--scheme", default="NVFP4", choices=["NVFP4", "NVFP4A16"],
                   help="NVFP4 = W4A4 (needs calibration, matches production); "
                        "NVFP4A16 = weight-only 4-bit, activations 16-bit")
    p.add_argument("--calib-dataset", default="ise-uiuc/Magicoder-Evol-Instruct-110K",
                   help="Code-instruction data (match the inference workload)")
    p.add_argument("--calib-samples", type=int, default=512)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--offload-dir", default=None,
                   help="Disk offload folder for the 480B if CPU RAM is tight")
    p.add_argument("--sequential-targets", nargs="+", default=None,
                   help="Granularity of the sequential pipeline. Default = the decoder layer, "
                        "which on a huge MoE (160 experts/layer) can OOM a single GPU. "
                        "Set to 'Linear' to calibrate ONE matmul at a time → far lower peak "
                        "VRAM (slower due to more on/off-loading). Fixes the "
                        "'choose a smaller module for sequential_targets' OOM.")
    return p.parse_args()


def build_calibration(tokenizer, args):
    """Code instructions, chat-template formatted to match the instruct model."""
    ds = load_dataset(args.calib_dataset, split="train")
    ds = ds.shuffle(seed=42).select(range(args.calib_samples))

    def to_text(ex):
        instr = ex.get("instruction") or ex.get("problem") or ex.get("prompt") or ""
        resp = ex.get("response") or ex.get("solution") or ex.get("completion") or ""
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": instr},
             {"role": "assistant", "content": resp}],
            tokenize=False,
        )
        return {"text": text}

    ds = ds.map(to_text, remove_columns=ds.column_names)

    def tokenize(ex):
        return tokenizer(ex["text"], truncation=True, max_length=args.seq_len,
                         add_special_tokens=False)  # chat template already added them

    return ds.map(tokenize, remove_columns=ds.column_names)


def main():
    args = parse_args()

    print(f"Loading {args.model_id} ...")
    load_kwargs = dict(dtype="auto", device_map="auto", trust_remote_code=True)
    if args.offload_dir:
        load_kwargs["offload_folder"] = args.offload_dir
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    # Match NVIDIA's exclusions: keep lm_head and every MoE router gate in high
    # precision (the gate decides routing — quantizing it wrecks expert selection).
    ignore = ["lm_head", "re:.*mlp.gate$"]
    print(f"Scheme: {args.scheme} | ignore (high precision): {ignore}")

    # sequential_targets controls how much is on the GPU at once during calibration.
    # Default (decoder layer) holds a whole MoE block (all experts) → OOM on big MoE.
    # "Linear" calibrates one matmul at a time → much lower peak VRAM.
    recipe = QuantizationModifier(
        targets="Linear", scheme=args.scheme, ignore=ignore,
        sequential_targets=args.sequential_targets,
    )

    # oneshot runs the sequential (layer-by-layer) pipeline; for NVFP4 (W4A4) it
    # uses the calibration data to fit activation scales. llm-compressor >=0.9
    # applies the MoE calibration context so all experts receive tokens.
    ds = None if args.scheme == "NVFP4A16" else build_calibration(tokenizer, args)
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=args.seq_len,
        num_calibration_samples=args.calib_samples,
    )

    print(f"Saving compressed-tensors NVFP4 model to {args.output_dir} ...")
    model.save_pretrained(args.output_dir, save_compressed=True)
    tokenizer.save_pretrained(args.output_dir)
    print("Done. Verify config.json shows quant_method 'compressed-tensors' and "
          "that q/k/v are quantized while lm_head / mlp.gate are not.")


if __name__ == "__main__":
    main()
