#!/usr/bin/env python3
"""
Memory-light offline RMSNorm fusion for Qwen3 — streams safetensors shards one
at a time, so peak RAM ~= one shard (~5 GB) instead of ~960 GB. Same math as
fuse_model.py: fold input_layernorm.weight (gamma) into q/k/v_proj, set gamma=1.

    python3 scripts/fuse_model_sharded.py \\
        --model-dir /workspace/qwen3-480b \\
        --output-dir /workspace/qwen3-480b-fused-bf16
"""
import argparse, json, os, re, shutil
import torch
from safetensors import safe_open
from safetensors.torch import save_file

QKV_RE   = re.compile(r"\.layers\.(\d+)\.self_attn\.(q|k|v)_proj\.weight$")
GAMMA_RE = re.compile(r"\.layers\.(\d+)\.input_layernorm\.weight$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    idx = os.path.join(args.model_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx) as f:
            shards = sorted(set(json.load(f)["weight_map"].values()))
    else:
        shards = ["model.safetensors"]

    # Pass 1: collect all gammas (tiny — one [h] vector per layer).
    gammas = {}
    for s in shards:
        with safe_open(os.path.join(args.model_dir, s), framework="pt") as f:
            for name in f.keys():
                m = GAMMA_RE.search(name)
                if m:
                    gammas[int(m.group(1))] = f.get_tensor(name)
    print(f"Found {len(gammas)} input_layernorm gammas")

    # Pass 2: rewrite each shard.
    folded = 0
    for s in shards:
        out = {}
        with safe_open(os.path.join(args.model_dir, s), framework="pt") as f:
            meta = f.metadata() or {}
            for name in f.keys():
                t = f.get_tensor(name)
                qkv = QKV_RE.search(name)
                if qkv:
                    layer = int(qkv.group(1))
                    if layer not in gammas:
                        raise KeyError(f"No gamma for layer {layer} ({name})")
                    t = (t * gammas[layer].to(t.dtype)).contiguous()  # fold gamma
                    folded += 1
                elif GAMMA_RE.search(name):
                    t = torch.ones_like(t)                            # reset gamma=1
                out[name] = t
        meta.setdefault("format", "pt")
        save_file(out, os.path.join(args.output_dir, s), metadata=meta)
        print(f"  wrote {s}  ({len(out)} tensors)")

    # Copy everything else (config, tokenizer, index json, ...).
    for fn in os.listdir(args.model_dir):
        src = os.path.join(args.model_dir, fn)
        if os.path.isfile(src) and not fn.endswith(".safetensors"):
            shutil.copy2(src, os.path.join(args.output_dir, fn))

    print(f"Done. Folded {folded} q/k/v projections -> {args.output_dir}")


if __name__ == "__main__":
    main()
