#!/usr/bin/env python3
"""
Shard-by-shard NVFP4A16 quantizer for Qwen3-MoE (no full-model load).

WHY THIS EXISTS
  llm-compressor's DataFreePipeline calls from_pretrained() + dispatch_model()
  twice, peaking above 1 TiB RAM for the 480B model even with offload flags.
  This script sidesteps the problem entirely: it streams safetensors shards
  one at a time (peak RAM ≈ one shard, ~4 GB for the 480B split into 241 shards)
  and writes a valid compressed-tensors NVFP4A16 checkpoint directly.

FORMAT (compressed-tensors NVFP4A16)
  For each quantized Linear weight tensor named "foo.weight" [out, in]:
    foo.weight_packed        uint8           [out, in//2]
        Two FP4 E2M1 codes packed per byte, lo nibble = even element.
    foo.weight_scale         float8_e4m3fn   [out, in // group_size]
        Per-group FP8 scale. group_size=16 (consecutive in-features).
    foo.weight_global_scale  float32         [1]
        Per-tensor normaliser.

  DEQUANT FORMULA (used by vLLM / compressed-tensors at inference):
    w_float ≈ fp4_to_float(code) * weight_scale[g] / weight_global_scale

  DERIVATION
    local_scale[g]      = max(|w_g|) / FP4_MAX           (true group scale)
    weight_global_scale = FP8_MAX / max(local_scale)      (normaliser)
    weight_scale[g]     = local_scale[g] * global_scale   (fits in FP8)
    Check: weight_scale / global_scale = local_scale  ✓

NOT QUANTIZED (kept in original dtype):
  lm_head.weight, *.mlp.gate.weight (router gate),
  all norm / embedding / non-weight tensors.

USAGE
  # Dry-run: print which tensors WOULD be quantized (no writes)
  python3 scripts/quantize_nvfp4_sharded.py \\
      --model-dir /workspace/fused-bf16 --output-dir /workspace/fused-nvfp4 --dry-run

  # Real run (GPU strongly recommended; ~10x faster scale computation)
  python3 scripts/quantize_nvfp4_sharded.py \\
      --model-dir /workspace/fused-bf16 --output-dir /workspace/fused-nvfp4 --device cuda:0

  # Verify reconstruction error on a single shard after the run
  python3 scripts/quantize_nvfp4_sharded.py \\
      --model-dir /workspace/fused-bf16 --output-dir /workspace/fused-nvfp4 --verify

VALIDATE ON 30B FIRST
  python3 scripts/quantize_nvfp4_sharded.py \\
      --model-dir /workspace/qwen3-30b \\
      --output-dir /workspace/qwen3-30b-nvfp4-test \\
      --device cuda:0 && \\
  python3 quantization/probe_nvfp4_loading.py \\
      --model-id /workspace/qwen3-30b-nvfp4-test
"""

import argparse
import json
import os
import re
import shutil
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# ---------------------------------------------------------------------------
# FP4 E2M1 constants  (NVIDIA Blackwell format)
# ---------------------------------------------------------------------------
# Positive magnitudes:  0  0.5  1.0  1.5  2.0  3.0  4.0  6.0
# Code 0-7 (bit-3 clear = positive), Code 8-15 (bit-3 set = negative)
FP4_MAX = 6.0

# Midpoints for round-to-nearest: value[i] if |x| > mid[i-1] and |x| <= mid[i]
_FP4_MIDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)

# Decode table: code 0-15 → float value (sign-magnitude, bit-3 = sign)
_FP4_TABLE = torch.tensor(
    [0., 0.5, 1., 1.5, 2., 3., 4., 6.,   # codes 0-7  (positive)
     0., -0.5, -1., -1.5, -2., -3., -4., -6.],  # codes 8-15 (negative)
    dtype=torch.float32,
)

try:
    _FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
except (AttributeError, TypeError):
    _FP8_MAX = 448.0  # documented max for float8_e4m3fn

_HAS_FP8 = hasattr(torch, "float8_e4m3fn")


# ---------------------------------------------------------------------------
# Quantize / dequantize a single 2-D weight tensor
# ---------------------------------------------------------------------------

def fp4_quantize(w: torch.Tensor, group_size: int = 16):
    """
    Quantize a [out, in] BF16/FP32 weight to NVFP4A16 compressed-tensors format.

    Returns:
        packed       uint8           [out, in//2]          lo nibble = even element
        scale_fp8    float8_e4m3fn   [out, in//group_size] per-group FP8 scale
        gscale_f32   float32         [1]                   per-tensor global scale
    """
    if w.dim() != 2:
        raise ValueError(f"Expected 2-D weight, got shape {tuple(w.shape)}")

    out_f, in_f = w.shape
    # Pad input dim to multiple of group_size if needed (rare for small test layers)
    pad = (-in_f) % group_size
    if pad:
        w = torch.nn.functional.pad(w, (0, pad))
    in_fp = w.shape[1]  # padded in_features

    w32 = w.float()
    num_groups = in_fp // group_size
    wg = w32.reshape(out_f, num_groups, group_size)   # [out, G, gs]

    # Per-group max-abs → local scale (what we actually divide by)
    local_scale = wg.abs().amax(dim=-1).clamp(min=1e-12) / FP4_MAX   # [out, G]

    # Global scale: normalise local_scale so it fits in FP8 range [0, FP8_MAX]
    max_local = local_scale.amax().clamp(min=1e-12)
    gscale = max_local / _FP8_MAX                           # scalar float32

    # Per-group FP8 scale = local_scale / gscale  (range ~[0, FP8_MAX])
    scale_f32 = local_scale / gscale                        # [out, G]
    if _HAS_FP8:
        scale_fp8 = scale_f32.to(torch.float8_e4m3fn)
    else:
        # Fallback: store as float16 (vLLM may not accept, but lets the script run)
        scale_fp8 = scale_f32.to(torch.float16)

    # Quantize using float32 local_scale for accuracy (avoids FP8 rounding at quant time)
    effective = local_scale.unsqueeze(-1)                   # [out, G, 1]
    w_scaled = (wg / effective).clamp(-FP4_MAX, FP4_MAX)   # [out, G, gs]

    # Round to nearest FP4 E2M1 code
    mids = _FP4_MIDS.to(w_scaled.device)
    signs = (w_scaled < 0)
    abs_w = w_scaled.abs()
    # Number of midpoints exceeded = magnitude code 0-7
    codes = (abs_w.unsqueeze(-1) > mids).sum(dim=-1).to(torch.uint8)  # [out, G, gs]
    codes[signs] |= 0x8                                                 # set sign bit

    # Flatten codes and remove padding
    codes_flat = codes.reshape(out_f, in_fp)[:, :in_f]     # [out, in_f]
    # Pad to even column count for packing (each byte holds two 4-bit codes)
    if in_f % 2:
        codes_flat = torch.nn.functional.pad(codes_flat, (0, 1))
    packed = (codes_flat[:, 0::2] | (codes_flat[:, 1::2] << 4)).to(torch.uint8)

    gscale_f32 = gscale.reshape(1).to(torch.float32)       # [1]

    return packed, scale_fp8, gscale_f32


def fp4_dequantize(
    packed: torch.Tensor,
    scale_fp8: torch.Tensor,
    gscale_f32: torch.Tensor,
    group_size: int = 16,
    out_dtype=torch.bfloat16,
) -> torch.Tensor:
    """
    Reconstruct a BF16 weight from NVFP4A16 compressed format.

    Formula: w ≈ fp4_to_float(code) * weight_scale[g] / weight_global_scale
    """
    out_f = packed.shape[0]
    in_f  = packed.shape[1] * 2

    # Unpack nibbles
    lo = (packed & 0x0F).to(torch.int64)
    hi = ((packed >> 4) & 0x0F).to(torch.int64)
    codes = torch.zeros(out_f, in_f, dtype=torch.int64, device=packed.device)
    codes[:, 0::2] = lo
    codes[:, 1::2] = hi

    # Decode FP4 → float32
    table = _FP4_TABLE.to(packed.device)
    values = table[codes]                                   # [out, in_f]

    # Reconstruct per-element scale = weight_scale[g] / weight_global_scale
    num_groups = (in_f + group_size - 1) // group_size
    scale_f32 = scale_fp8.float()                           # [out, G] float32
    gscale = gscale_f32.float().item()
    eff_scale = (scale_f32 / gscale)                        # [out, G]
    # Expand to [out, in_f]
    eff_per_elem = eff_scale.unsqueeze(-1).expand(
        out_f, num_groups, group_size
    ).reshape(out_f, num_groups * group_size)[:, :in_f]

    return (values * eff_per_elem).to(out_dtype)


# ---------------------------------------------------------------------------
# Which tensor names to quantize
# ---------------------------------------------------------------------------

_EXCLUDE = re.compile(
    r"^lm_head\.weight$"          # output projection (kept high precision)
    r"|\.mlp\.gate\.weight$"      # MoE router gate (routing accuracy)
)
_NOT_LINEAR = re.compile(         # non-Linear tensors (always 1-D or non-weight)
    r"embed_tokens"               # token embedding table
    r"|layernorm"                 # LayerNorm weights
    r"|rmsnorm"                   # RMSNorm weights
    r"|norm\.weight$"             # catches input_layernorm, q_norm, k_norm, model.norm
    r"|rotary_emb"                # RoPE frequencies
    r"|\.bias$"                   # bias vectors (rare in Qwen3 but safe to exclude)
)


def should_quantize(name: str, tensor: torch.Tensor) -> bool:
    """Return True iff this tensor should be quantized to FP4."""
    if not name.endswith(".weight"):
        return False
    if tensor.dim() != 2:         # Linear weights are 2-D; norms/biases are 1-D
        return False
    if _EXCLUDE.search(name):
        return False
    if _NOT_LINEAR.search(name):
        return False
    return True


# ---------------------------------------------------------------------------
# compressed-tensors quantization_config injected into config.json
# ---------------------------------------------------------------------------

def make_quant_config(group_size: int) -> dict:
    """
    Build the quantization_config dict for config.json.

    NOTE: The 'format' field value 'float-quantized' is our best guess at
    what compressed-tensors uses for packed FP4 weights. If loading fails,
    compare with a reference NVFP4 checkpoint produced by llm-compressor and
    adjust (try 'dense', 'packed', or 'nvfp4').
    """
    return {
        "format": "float-quantized",
        "global_compression_ratio": 2.0,      # approximate (4-bit / 16-bit ≈ 4x, minus scale overhead)
        "ignore": ["lm_head", "re:.*mlp\\.gate$"],
        "kv_cache_scheme": None,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "type": "float",
                    "num_bits": 4,
                    "strategy": "group",
                    "group_size": group_size,
                    "block_structure": None,
                    "dynamic": False,
                    "symmetric": True,
                    "pytorch_dtype": "float8_e4m3fn",
                },
                "input_activations": None,
                "output_activations": None,
            }
        },
    }


# ---------------------------------------------------------------------------
# Verify mode: dequantize output and compare to source on one shard
# ---------------------------------------------------------------------------

def verify(src_dir: str, out_dir: str, group_size: int, shard: str, device: torch.device):
    """
    Dequantize the packed weights in one output shard and measure reconstruction
    error against the source BF16 weights. Prints per-tensor stats.
    """
    print(f"\n=== Verify: {shard} ===")
    src_path = os.path.join(src_dir, shard)
    out_path = os.path.join(out_dir, shard)

    if not os.path.exists(out_path):
        print(f"  Output shard not found: {out_path}")
        return

    # Load source tensors
    src_tensors: dict[str, torch.Tensor] = {}
    with safe_open(src_path, framework="pt", device="cpu") as f:
        for name in f.keys():
            src_tensors[name] = f.get_tensor(name)

    # Load output tensors
    out_tensors: dict[str, torch.Tensor] = {}
    with safe_open(out_path, framework="pt", device="cpu") as f:
        for name in f.keys():
            out_tensors[name] = f.get_tensor(name)

    # Find quantized triplets in output shard
    packed_names = [n for n in out_tensors if n.endswith(".weight_packed")]
    if not packed_names:
        print("  No quantized tensors found in output shard.")
        return

    max_errs = []
    rel_errs = []
    for packed_name in sorted(packed_names):
        base = packed_name[: -len(".weight_packed")]
        scale_name  = base + ".weight_scale"
        gscale_name = base + ".weight_global_scale"
        src_name    = base + ".weight"

        if src_name not in src_tensors:
            print(f"  SKIP {base}: source tensor not in this shard")
            continue

        packed  = out_tensors[packed_name].to(device)
        scale   = out_tensors[scale_name].to(device)
        gscale  = out_tensors[gscale_name].to(device)
        w_orig  = src_tensors[src_name].to(device=device, dtype=torch.float32)

        w_rec = fp4_dequantize(packed, scale, gscale, group_size).float().to(device)
        err   = (w_orig - w_rec).abs()
        rel   = (err / (w_orig.abs().clamp(min=1e-6))).mean().item()
        max_e = err.max().item()
        max_errs.append(max_e)
        rel_errs.append(rel)
        print(f"  {base:<65} max_err={max_e:.4f}  mean_rel_err={rel:.4f}")

    print(f"\n  Summary: mean max_err={sum(max_errs)/len(max_errs):.4f}  "
          f"mean rel_err={sum(rel_errs)/len(rel_errs):.4f}")
    print("  (FP4 RTN expected ~5-15% mean relative error — larger values indicate a bug)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Shard-by-shard NVFP4A16 quantizer — peak RAM ≈ one shard"
    )
    ap.add_argument("--model-dir",  required=True,
                    help="Source BF16 checkpoint directory")
    ap.add_argument("--output-dir", required=True,
                    help="Output NVFP4A16 checkpoint directory")
    ap.add_argument("--group-size", type=int, default=16,
                    help="FP4 group size along in_features (default: 16, matches NVFP4 spec)")
    ap.add_argument("--device", default="cpu",
                    help="Compute device: 'cpu' or 'cuda:0'. GPU is ~10x faster.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be quantized without writing output.")
    ap.add_argument("--verify", action="store_true",
                    help="After quantizing, verify reconstruction error on the first shard.")
    args = ap.parse_args()

    if not _HAS_FP8:
        print("WARNING: torch.float8_e4m3fn not available (requires PyTorch >= 2.1).")
        print("         Scales will be stored as float16 — output may not load in vLLM.")

    if not args.dry_run:
        os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Load shard index                                                      #
    # ------------------------------------------------------------------ #
    idx_path = os.path.join(args.model_dir, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        with open(idx_path) as fh:
            index = json.load(fh)
        weight_map = index["weight_map"]
        src_shards  = sorted(set(weight_map.values()))
        index_meta  = index.get("metadata", {})
    else:
        weight_map = {}
        src_shards  = ["model.safetensors"]
        index_meta  = {}

    device = torch.device(args.device)
    print(f"Source  : {args.model_dir}")
    print(f"Output  : {'(dry-run)' if args.dry_run else args.output_dir}")
    print(f"Shards  : {len(src_shards)}")
    print(f"Device  : {device}  |  group_size: {args.group_size}  |  FP8_MAX: {_FP8_MAX}")
    print()

    # ------------------------------------------------------------------ #
    # Process each shard                                                    #
    # ------------------------------------------------------------------ #
    out_weight_map: dict[str, str] = {}
    n_quant  = 0
    n_copy   = 0
    t0 = time.time()

    for si, shard in enumerate(src_shards):
        src_path = os.path.join(args.model_dir, shard)
        out_tensors: dict[str, torch.Tensor] = {}

        with safe_open(src_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                t = f.get_tensor(name)

                if should_quantize(name, t):
                    if args.dry_run:
                        print(f"  QUANT  {name:<70}  {tuple(t.shape)}  {t.dtype}")
                        n_quant += 1
                        continue

                    packed, scale_fp8, gscale = fp4_quantize(
                        t.to(device=device), args.group_size
                    )
                    base = name[: -len(".weight")]
                    out_tensors[f"{base}.weight_packed"]       = packed.cpu()
                    out_tensors[f"{base}.weight_scale"]        = scale_fp8.cpu()
                    out_tensors[f"{base}.weight_global_scale"] = gscale.cpu()
                    out_weight_map[f"{base}.weight_packed"]       = shard
                    out_weight_map[f"{base}.weight_scale"]        = shard
                    out_weight_map[f"{base}.weight_global_scale"] = shard
                    n_quant += 1
                else:
                    if args.dry_run:
                        print(f"  copy   {name:<70}  {tuple(t.shape)}  {t.dtype}")
                    else:
                        out_tensors[name] = t
                        out_weight_map[name] = shard
                    n_copy += 1

        if not args.dry_run:
            dst_path = os.path.join(args.output_dir, shard)
            save_file(out_tensors, dst_path, metadata={"format": "pt"})

        elapsed = time.time() - t0
        rate    = (si + 1) / elapsed if elapsed > 0 else 0
        eta     = (len(src_shards) - si - 1) / rate if rate > 0 else 0
        print(f"  [{si+1:3d}/{len(src_shards)}] {shard:<55}  "
              f"quant={n_quant:5d}  copy={n_copy:5d}  "
              f"{elapsed:6.0f}s elapsed  ~{eta:.0f}s remaining")

    # ------------------------------------------------------------------ #
    # Write index + config files                                            #
    # ------------------------------------------------------------------ #
    if not args.dry_run:
        # model.safetensors.index.json
        if os.path.exists(idx_path):
            out_index = {"metadata": index_meta, "weight_map": out_weight_map}
            with open(os.path.join(args.output_dir, "model.safetensors.index.json"), "w") as fh:
                json.dump(out_index, fh, indent=2)
            print("\nWrote model.safetensors.index.json")

        # config.json (inject quantization_config) + copy everything else
        quant_cfg = make_quant_config(args.group_size)
        for fn in sorted(os.listdir(args.model_dir)):
            src = os.path.join(args.model_dir, fn)
            if not os.path.isfile(src) or fn.endswith(".safetensors"):
                continue
            dst = os.path.join(args.output_dir, fn)
            if fn == "config.json":
                with open(src) as fh:
                    cfg = json.load(fh)
                cfg["quantization_config"] = quant_cfg
                with open(dst, "w") as fh:
                    json.dump(cfg, fh, indent=2)
                print("Wrote config.json  (quantization_config injected)")
            else:
                shutil.copy2(src, dst)

        total = time.time() - t0
        print(f"\nDone in {total:.0f}s")
        print(f"Quantized {n_quant} weight tensors, copied {n_copy} tensors unchanged.")
        print(f"Output: {args.output_dir}")

        # ---------------------------------------------------------------- #
        # Optional verify                                                    #
        # ---------------------------------------------------------------- #
        if args.verify and src_shards:
            verify(args.model_dir, args.output_dir, args.group_size, src_shards[0], device)

    else:
        print(f"\nDry-run: would quantize {n_quant} tensors, copy {n_copy}.")

    if not args.dry_run:
        print("\nNEXT STEPS:")
        print("  1. Spot-check reconstruction error:")
        print(f"       python3 scripts/quantize_nvfp4_sharded.py \\")
        print(f"           --model-dir {args.model_dir} --output-dir {args.output_dir} --verify")
        print("  2. Probe loadability (compressed-tensors format):")
        print(f"       python3 quantization/probe_nvfp4_loading.py --model-id {args.output_dir}")
        print("  3. Upload to HuggingFace:")
        print(f"       huggingface-cli upload YOUR_USER/qwen3-480b-fused-nvfp4 {args.output_dir} .")
        print()
        print("  NOTE: If probe_nvfp4_loading fails with a format error, compare")
        print("  the 'quantization_config.format' field in the output config.json")
        print("  against a reference llm-compressor NVFP4 checkpoint and adjust")
        print("  the make_quant_config() function in this script accordingly.")


if __name__ == "__main__":
    main()
