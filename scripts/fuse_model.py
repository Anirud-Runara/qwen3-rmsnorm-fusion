#!/usr/bin/env python3
"""
Offline in-place RMSNorm fusion for Qwen3-Coder-480B-A35B-Instruct.

What this script does:
  1. Loads the BF16 model onto CPU (requires ~900 GB RAM).
  2. For every decoder layer, absorbs input_layernorm.weight (gamma) directly
     into q_proj, k_proj, v_proj weight matrices:
         W_new = W * gamma    (element-wise row-wise scale, shape [out, h])
     Then resets input_layernorm.weight to ones (identity, so any residual
     layernorm calls produce x*1 = x, which is correct since the scale is
     already absorbed).
  3. Saves the modified model to --output-dir using save_pretrained().

This offline transform is equivalent to the runtime patch in patch_qwen3.py,
but bakes the weights permanently into the checkpoint so downstream consumers
see the fused weights without needing the runtime patch at all.

Usage:
    python3 scripts/fuse_model.py \\
        --model-id Qwen/Qwen3-Coder-480B-A35B-Instruct \\
        --output-dir /workspace/qwen3-480b-fused-bf16

Optional flags:
    --local-model-dir   Path to pre-downloaded weights (skips HF download).
    --sanity-check      After saving, reload and run a single generation to
                        confirm the output is coherent (requires 1+ GPU).
    --max-shard-size    Shard size for save_pretrained (default: "10GB").
"""

import argparse
import gc
import os
import sys
import time


def parse_args():
    p = argparse.ArgumentParser(description="Offline Qwen3-480B RMSNorm fusion")
    p.add_argument(
        "--model-id",
        default="Qwen/Qwen3-Coder-480B-A35B-Instruct",
        help="HuggingFace model ID or local path",
    )
    p.add_argument(
        "--local-model-dir",
        default=None,
        help="If set, load from this local directory instead of HF Hub",
    )
    p.add_argument(
        "--output-dir",
        default="qwen3-480b-fused-bf16",
        help="Directory to save the fused model",
    )
    p.add_argument(
        "--max-shard-size",
        default="10GB",
        help="Shard size passed to save_pretrained (default: 10GB)",
    )
    p.add_argument(
        "--sanity-check",
        action="store_true",
        help="After saving, reload fused model on GPU and run a short generation",
    )
    return p.parse_args()


def fuse_layer(layer, layer_idx: int) -> None:
    """
    In-place: absorb input_layernorm.weight into q/k/v_proj, then reset norm to ones.

    The math: for RMSNorm with scale gamma:
        RMSNorm(x) * W = (x / rms(x)) * gamma * W
                       = (x / rms(x)) * (W * gamma)    <- new weight matrix
    So W_new[i, :] = W[i, :] * gamma[:]  (row i of W scaled by gamma)
    """
    attn = layer.self_attn
    gamma = layer.input_layernorm.weight.data  # [h]

    for proj in (attn.q_proj, attn.k_proj, attn.v_proj):
        # proj.weight shape: [out_features, in_features] = [out, h]
        # gamma broadcast: [h] -> multiply each row of W by the corresponding gamma
        proj.weight.data.mul_(gamma)

    # Reset norm weight to ones: any code that still calls input_layernorm will
    # now compute x * 1 = x (identity scale), which is correct because the norm
    # scale is already absorbed into the projection weights.
    layer.input_layernorm.weight.data.fill_(1.0)


def main():
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = args.local_model_dir if args.local_model_dir else args.model_id
    print(f"Loading model from: {model_path}")
    print("(This requires ~900 GB RAM and will take several minutes)")

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    n_layers = len(model.model.layers)
    print(f"\nFusing RMSNorm into QKV projections for {n_layers} decoder layers ...")

    t1 = time.time()
    for layer_idx, layer in enumerate(model.model.layers):
        fuse_layer(layer, layer_idx)
        if (layer_idx + 1) % 10 == 0 or layer_idx == n_layers - 1:
            elapsed = time.time() - t1
            rate = (layer_idx + 1) / elapsed
            remaining = (n_layers - layer_idx - 1) / rate if rate > 0 else 0
            print(
                f"  Layer {layer_idx+1:4d}/{n_layers}  "
                f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)"
            )

    print(f"\nFusion complete in {time.time() - t1:.1f}s")

    # Force a GC pass before saving to free any intermediate tensors
    gc.collect()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nSaving fused model to: {args.output_dir}")
    print(f"(Shard size: {args.max_shard_size})")

    t2 = time.time()
    model.save_pretrained(args.output_dir, max_shard_size=args.max_shard_size)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved in {time.time() - t2:.1f}s")

    # ------------------------------------------------------------------
    # Optional sanity check: reload on GPU and run a short generation
    # ------------------------------------------------------------------
    if args.sanity_check:
        print("\nRunning sanity check (loading fused model on GPU) ...")
        if not torch.cuda.is_available():
            print("  [SKIP] No CUDA device available for sanity check.")
        else:
            del model
            gc.collect()
            torch.cuda.empty_cache()

            fused_model = AutoModelForCausalLM.from_pretrained(
                args.output_dir,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            fused_model.eval()
            fused_tokenizer = AutoTokenizer.from_pretrained(args.output_dir, trust_remote_code=True)

            prompt = "def quicksort(arr):"
            inputs = fused_tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = fused_model.generate(**inputs, max_new_tokens=64, do_sample=False)
            generated = fused_tokenizer.decode(out[0], skip_special_tokens=True)
            print(f"\n  Prompt : {prompt!r}")
            print(f"  Output : {generated!r}")
            print("\n  Sanity check complete. Verify the output looks like valid Python code.")

    print(f"\nTotal time: {time.time() - t0:.1f}s")
    print(f"Fused BF16 model is ready at: {args.output_dir}")


if __name__ == "__main__":
    main()
