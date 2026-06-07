"""
Benchmark: RMSNorm+QKV Fusion vs Non-Fused (Qwen3 / Qwen3-MoE NVFP4)
PyTorch + CUDA

WHAT IS BENCHMARKED
    The fusion site: input_layernorm → [q_proj, k_proj, v_proj]

    Non-fused: norm(x) written to HBM, then read back for each of the three
               separate Q/K/V matmuls (3 sequential kernel launches).
    Fused:     single combined matmul (one kernel) + CUDA RMSNorm+normalize
               kernel (one kernel). Eliminates the intermediate HBM round-trip.

MODES
    runtime-patch (default, recommended)
        Loads the non-fused checkpoint once. Builds both bench modules from the
        same layer weights — nonfused runs norm → q/k/v separately; fused runs
        FusedRMSNormCombinedLinear CUDA kernel (V1 or V3). Cleanest comparison:
        the measured delta is attributable purely to the fusion kernel, not to
        weight format or quantization differences.

    checkpoints
        Loads models/non-fused/ and models/fused/ from separate checkpoint
        directories. Use when you want to compare the offline-fused checkpoint
        (γ absorbed into weights) + CUDA kernel vs the plain unfused baseline.

USAGE
    # Discover the layer path (reads index only — no weight load):
    python benchmarks/benchmark_rmsnorm_linear_fusion.py \\
        --dir /workspace --print-keys

    # Smoke test (load + one forward, then exit):
    python benchmarks/benchmark_rmsnorm_linear_fusion.py \\
        --dir /workspace --test-load --variant V3

    # Full benchmark sweep, V3 kernel:
    python benchmarks/benchmark_rmsnorm_linear_fusion.py \\
        --dir /workspace --variant V3

    # Full benchmark sweep, V1 kernel (compare with V3 results):
    python benchmarks/benchmark_rmsnorm_linear_fusion.py \\
        --dir /workspace --variant V1

DIRECTORY LAYOUT
    <dir>/
      models/
        non-fused/   — unfused Qwen3 NVFP4 checkpoint (required for runtime-patch)
        fused/       — offline-fused checkpoint (required for checkpoints mode only)

REQUIREMENTS
    pip install torch transformers safetensors
    pip install compressed-tensors   # for NVFP4 checkpoints
    # Build the CUDA extension first: pip install -e . (from repo root)
"""

import argparse
import copy
import gc
import json
import os
import sys
import time
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Repo root on path so we can import src.*
_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_BENCH_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEVICE = "cuda"
# Qwen3 NVFP4 checkpoints use BF16 activations; FP16 causes dtype errors with
# compressed-tensors matmul kernels.
DTYPE = torch.bfloat16

# Shape sweep: (batch_size, seq_len, hidden_dim, out_dim)
# hidden_dim and out_dim are overridden at runtime from actual model weights.
SHAPE_SWEEP: List[Tuple[int, int, int, int]] = [
    (1,   128,  4096, 4096),
    (1,   512,  4096, 4096),
    (1,  2048,  4096, 4096),
    (8,   128,  4096, 4096),
    (8,   512,  4096, 4096),
    (8,  2048,  4096, 4096),
    (32,  128,  4096, 4096),
    (32,  512,  4096, 4096),
    (32, 2048,  4096, 4096),
]

WARMUP_ITERS    = 50
MEASURE_ITERS   = 200
NUMERICAL_ITERS = 10


# ---------------------------------------------------------------------------
# Benchmark modules
# ---------------------------------------------------------------------------

# E2M1 (FP4) magnitude lookup: the 3 magnitude bits index these values.
# The 4th bit is the sign.  This table is fixed by the NVFP4 spec.
_E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)


def _dequantize_nvfp4(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
    out_dtype: torch.dtype = DTYPE,
) -> torch.Tensor:
    """
    Reconstruct a dense [out, in] weight from NVFP4 compressed-tensors buffers.

    NVFP4 layout (fixed by the spec, identical across compressed-tensors
    versions):

        weight_packed        uint8           (out, in // 2)   2 FP4 codes/byte
        weight_scale         float8_e4m3fn   (out, in // 16)  per-block scale
        weight_global_scale  float32         scalar           per-tensor scale

    Reconstruction, per element:

        W = E2M1(code) * (weight_scale / weight_global_scale)

    The global scale stores the *quantization* scale, so we divide by it
    (NVIDIA/compressed-tensors convention).  Block size is inferred from the
    shapes (in // n_scale_cols), which is 16 for NVFP4.
    """
    device = weight_packed.device
    lut = _E2M1_LUT.to(device)

    packed = weight_packed.view(torch.uint8)            # (out, in//2)
    out_dim, in_half = packed.shape

    low_nib = packed & 0x0F
    high_nib = (packed >> 4) & 0x0F

    def _decode(nib: torch.Tensor) -> torch.Tensor:
        sign = 1.0 - 2.0 * ((nib >> 3) & 1).to(torch.float32)  # bit3 → sign
        return sign * lut[(nib & 0x07).long()]

    # Low nibble = even column, high nibble = odd column.
    vals = torch.empty(out_dim, in_half * 2, device=device, dtype=torch.float32)
    vals[:, 0::2] = _decode(low_nib)
    vals[:, 1::2] = _decode(high_nib)

    in_dim = in_half * 2
    n_blocks = weight_scale.shape[1]
    block = in_dim // n_blocks                                  # == 16

    scale = weight_scale.to(torch.float32) / weight_global_scale.to(torch.float32)
    scale = scale.repeat_interleave(block, dim=1)               # (out, in)

    return (vals * scale).to(out_dtype)


def _decompress_weight(linear: nn.Module) -> torch.Tensor:
    """
    Return a dense [out, in] weight tensor.

    Only two cases exist — and both are read directly from the module's own
    tensors, with no dependency on any compressed-tensors decompression API:

      1. Dense weight present  → return it (plain BF16 checkpoint, or a weight
         that was restored after CompressedLinear wrapping displaced it).
      2. NVFP4 packed buffers  → dequantize with _dequantize_nvfp4().

    `getattr(linear, "weight", None)` safely returns None whether .weight is
    set to None or removed from _parameters entirely by the quantizer hooks.
    """
    w = getattr(linear, "weight", None)
    if w is not None:
        return w.data

    wp = getattr(linear, "weight_packed", None)
    if wp is not None:
        return _dequantize_nvfp4(
            wp, linear.weight_scale, linear.weight_global_scale, DTYPE
        )

    raise AttributeError(
        f"{type(linear).__name__} has neither a dense .weight nor NVFP4 "
        ".weight_packed buffers. The checkpoint did not load correctly — "
        "check for 'WARNING: quantizer setup failed' or missing-key warnings "
        "earlier in the output."
    )


def _make_dense_linear(linear: nn.Module, dtype: torch.dtype) -> nn.Linear:
    """Build a plain nn.Linear with dense weights decompressed from `linear`."""
    W = _decompress_weight(linear).to(dtype)
    out_dim, in_dim = W.shape
    b = linear.bias.data.to(dtype) if linear.bias is not None else None
    dense = nn.Linear(in_dim, out_dim, bias=(b is not None),
                      device=W.device, dtype=dtype)
    dense.weight.data.copy_(W)
    if b is not None:
        dense.bias.data.copy_(b)
    return dense


class _UnfusedNormQKV(nn.Module):
    """
    Non-fused baseline: RMSNorm → q_proj, k_proj, v_proj run separately.

    All three projections are dense BF16 nn.Linear modules (decompressed from
    NVFP4 at build time) so the comparison with the fused kernel is purely
    about kernel-level efficiency, not arithmetic precision.

    forward(x) → [batch, seq, q_dim + k_dim + v_dim]
    """

    def __init__(self, norm: nn.Module, q: nn.Linear, k: nn.Linear, v: nn.Linear):
        super().__init__()
        self.norm = norm
        self.q = q
        self.k = k
        self.v = v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm(x)
        return torch.cat([self.q(normed), self.k(normed), self.v(normed)], dim=-1)


class _FusedNormQKVWrapper(nn.Module):
    """
    Fused benchmark module: wraps FusedRMSNormCombinedLinear*.

    forward(x) → [batch, seq, q_dim + k_dim + v_dim]  (concatenated splits)
    """

    def __init__(self, fused_mod: nn.Module):
        super().__init__()
        self.fused = fused_mod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = self.fused(x)  # list of tensors
        return torch.cat(parts, dim=-1)


def _build_unfused_bench(layer: nn.Module, device: str) -> _UnfusedNormQKV:
    """
    Build the non-fused benchmark module from a decoder layer.

    Decompresses q/k/v_proj to dense BF16 nn.Linear so the only difference
    between this and the fused bench is the kernel path, not weight format.
    """
    norm   = layer.input_layernorm
    attn   = layer.self_attn
    q_lin  = _make_dense_linear(attn.q_proj, DTYPE)
    k_lin  = _make_dense_linear(attn.k_proj, DTYPE)
    v_lin  = _make_dense_linear(attn.v_proj, DTYPE)
    return _UnfusedNormQKV(norm, q_lin, k_lin, v_lin).to(device)


def _build_fused_bench(layer: nn.Module, variant: str, device: str) -> _FusedNormQKVWrapper:
    """
    Build the fused benchmark module from a decoder layer.

    Decompresses q/k/v_proj, absorbs gamma into the combined weight matrix,
    and wraps a FusedRMSNormCombinedLinearV1/V3 CUDA module.
    """
    from src.weight_transform import compute_fused_weights_rmsnorm_combined
    from src.fused_forward import FusedRMSNormCombinedLinearV1, FusedRMSNormCombinedLinearV3

    _VARIANT_CLS = {
        "V1": FusedRMSNormCombinedLinearV1,
        "V3": FusedRMSNormCombinedLinearV3,
    }
    cls = _VARIANT_CLS[variant]

    attn = layer.self_attn
    norm = layer.input_layernorm
    W_combined, b_combined, split_sizes, h, eps = compute_fused_weights_rmsnorm_combined(
        norm, [attn.q_proj, attn.k_proj, attn.v_proj]
    )
    fused_mod = cls(
        W_combined.to(device, dtype=DTYPE),
        b_combined.to(device, dtype=DTYPE),
        split_sizes,
        h,
        eps,
    )
    return _FusedNormQKVWrapper(fused_mod)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _get_nested_attr(obj, dot_path: str):
    """Navigate 'model.layers.0.self_attn' → obj.model.layers[0].self_attn."""
    for part in dot_path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def _parse_layer_idx(layer_path: str) -> int:
    parts = layer_path.split(".")
    if "layers" not in parts:
        raise ValueError(f"--layer-path must contain 'layers.<idx>' (got {layer_path!r})")
    return int(parts[parts.index("layers") + 1])


def _gpu_max_memory(num_gpus: Optional[int] = None, reserve_frac: float = 0.08) -> dict:
    """Per-GPU memory budget for device_map='auto'. Override via env vars."""
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    if os.environ.get("BENCHMARK_GPU_MEM"):
        mem = {i: os.environ["BENCHMARK_GPU_MEM"] for i in range(num_gpus)}
    else:
        mem = {}
        for i in range(num_gpus):
            total_gib = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            usable = max(1, int(total_gib * (1.0 - reserve_frac)))
            mem[i] = f"{usable}GiB"
    mem["cpu"] = os.environ.get("BENCHMARK_CPU_MEM", "128GiB")
    return mem


def _print_model_device_map(model: nn.Module, label: str) -> None:
    hf_map = getattr(model, "hf_device_map", None)
    if hf_map:
        counts: dict = {}
        for dev in hf_map.values():
            counts[str(dev)] = counts.get(str(dev), 0) + 1
        print(f"  [{label}] hf_device_map ({len(hf_map)} modules): {dict(sorted(counts.items()))}")
    else:
        counts = {}
        for p in model.parameters():
            d = str(p.device)
            counts[d] = counts.get(d, 0) + 1
        print(f"  [{label}] parameter devices: {dict(sorted(counts.items()))}")


def _normalize_quant_config(q_cfg) -> dict:
    """
    Return a sanitized quantization_config dict compatible with the installed
    compressed-tensors pydantic schema.

    Older checkpoints (saved with compressed-tensors < 0.8) may include:
      - weights.pytorch_dtype  — removed in newer pydantic schema
      - config_groups values as dicts when a list is expected

    We strip the known-offending fields so AutoHfQuantizer can parse the rest.
    """
    import copy
    q: dict = q_cfg if isinstance(q_cfg, dict) else (
        q_cfg.to_dict() if hasattr(q_cfg, "to_dict") else dict(q_cfg)
    )
    q = copy.deepcopy(q)

    def _clean_quant_scheme(scheme: dict) -> dict:
        for sub in ("weights", "input_activations", "output_activations"):
            if isinstance(scheme.get(sub), dict):
                scheme[sub].pop("pytorch_dtype", None)
        return scheme

    groups = q.get("config_groups", {})
    if isinstance(groups, dict):
        for v in groups.values():
            if isinstance(v, dict):
                _clean_quant_scheme(v)
    elif isinstance(groups, list):
        for item in groups:
            if isinstance(item, dict):
                _clean_quant_scheme(item)

    return q


def _load_layer_gpu(
    model_dir: str,
    layer_path: str,
    label: str,
    device: str = "cuda:0",
) -> nn.Module:
    """
    Load only the Qwen3 decoder layer at `layer_path` onto `device`.

    Skips loading the full ~480B model; reads only the ~20 tensors for one
    decoder layer directly from the relevant safetensors shards. Applies
    compressed-tensors NVFP4 quantizer wrappers when the checkpoint has a
    quantization_config.

    Falls back gracefully: if the decoder layer class cannot be imported,
    raises RuntimeError with a suggestion to use --load-mode full.
    """
    from safetensors.torch import load_file
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    t0 = time.perf_counter()
    print(f"\nLoading {label} (layer-only) from {model_dir} ...")
    print(f"  layer-path: {layer_path}  device: {device}")

    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    text_cfg = getattr(config, "text_config", config)

    layer_idx = _parse_layer_idx(layer_path)

    # Try Qwen3-MoE first, then dense Qwen3
    decoder_cls = None
    for class_ref in (
        "modeling_qwen3_moe.Qwen3MoeDecoderLayer",
        "modeling_qwen3.Qwen3DecoderLayer",
    ):
        try:
            decoder_cls = get_class_from_dynamic_module(
                class_ref, model_dir, trust_remote_code=True
            )
            break
        except Exception:
            continue

    # Fallback: import decoder layer class directly from transformers (no
    # custom modeling files needed in the checkpoint directory).
    if decoder_cls is None:
        for mod_path, cls_name in (
            ("transformers.models.qwen3_moe.modeling_qwen3_moe", "Qwen3MoeDecoderLayer"),
            ("transformers.models.qwen3.modeling_qwen3",         "Qwen3DecoderLayer"),
        ):
            try:
                import importlib
                decoder_cls = getattr(importlib.import_module(mod_path), cls_name)
                print(f"  Using {cls_name} from {mod_path}")
                break
            except Exception:
                continue

    if decoder_cls is None:
        raise RuntimeError(
            f"Could not import Qwen3 decoder layer class from {model_dir} "
            "or from the installed transformers package.\n"
            "Ensure transformers >= 4.51 is installed, or retry with --load-mode full."
        )

    module = decoder_cls(text_cfg, layer_idx=layer_idx)
    module = module.to(device=device, dtype=DTYPE)

    # Apply NVFP4 quantizer wrappers (wraps nn.Linear → CompressedLinear).
    # Falls back to a schema-normalized config when the checkpoint was saved
    # with an older compressed-tensors version whose pydantic fields differ.
    q_cfg = getattr(config, "quantization_config", None)
    quantizer = None
    if q_cfg is not None:
        from transformers.quantizers.auto import AutoHfQuantizer
        for attempt, cfg_src in enumerate((q_cfg, _normalize_quant_config(q_cfg))):
            try:
                quantizer = AutoHfQuantizer.from_config(cfg_src)
                quantizer._process_model_before_weight_loading(module)
                tag = "" if attempt == 0 else " (normalized config)"
                print(f"  NVFP4 quantizer wrappers applied{tag} ({time.perf_counter() - t0:.1f}s)")
                break
            except Exception as e:
                if attempt == 0:
                    print(f"  INFO: quantizer setup failed on raw config ({type(e).__name__}), "
                          "retrying with normalized config …")
                else:
                    print(f"  WARNING: quantizer setup failed on both raw and normalized configs "
                          f"({e}); weights will be loaded as plain BF16.\n"
                          "  If the fused checkpoint is NVFP4, q/k/v weights may not load "
                          "correctly — verify compressed-tensors version matches the one "
                          "used to create the checkpoint.")
                    q_cfg = None
                    quantizer = None

    # Load weights from safetensors shards
    prefix = layer_path + "."
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    layer_keys = [k for k in weight_map if k.startswith(prefix)]
    if not layer_keys:
        raise ValueError(f"No weights found under prefix {prefix!r} in {index_path}")

    shards_needed = {weight_map[k] for k in layer_keys}
    state: dict[str, torch.Tensor] = {}
    _prefix_mismatch_warned = False
    for shard in shards_needed:
        raw = load_file(os.path.join(model_dir, shard), device=device)
        # Build a secondary lookup that strips the leading 'model.' scope from
        # each key.  Some quantization / fusion pipelines save shard files
        # without the top-level 'model.' prefix even though the index has it,
        # causing an exact-match miss for every tensor except the few whose
        # names happen to be identical in both the index and the shard file.
        alt = {k2.removeprefix("model."): v for k2, v in raw.items()}
        for k in layer_keys:
            if k in raw:
                state[k[len(prefix):]] = raw[k]
            else:
                k_alt = k.removeprefix("model.")
                if k_alt in alt:
                    if not _prefix_mismatch_warned:
                        print(
                            "  NOTE: shard keys lack 'model.' prefix — "
                            "loading via stripped-prefix fallback."
                        )
                        _prefix_mismatch_warned = True
                    state[k[len(prefix):]] = alt[k_alt]

    missing, unexpected = module.load_state_dict(state, strict=False)
    if missing:
        print(f"  WARNING: {len(missing)} missing keys: {missing[:4]}{'...' if len(missing) > 4 else ''}")
    if unexpected:
        print(f"  WARNING: {len(unexpected)} unexpected keys: {unexpected[:4]}{'...' if len(unexpected) > 4 else ''}")

    if q_cfg is not None:
        try:
            quantizer._process_model_after_weight_loading(module)
        except Exception as e:
            print(f"  WARNING: post-weight-loading quantizer step failed ({e})")

    # ── Restore plain BF16 weights that CompressedLinear wrapping displaced ──
    # When a checkpoint uses NVFP4 only for certain layers (e.g., MoE experts
    # but NOT attention projections), _process_model_before_weight_loading
    # still wraps ALL linears as CompressedLinear and clears their .weight.
    # The plain 'weight' tensors in the state dict then fall into unexpected_keys
    # and never load.  Detect these and restore them directly.
    _restored: list = []
    for mod_name, submod in module.named_modules():
        if getattr(submod, "weight", None) is not None:
            continue  # weight already loaded — nothing to fix
        plain_key = f"{mod_name}.weight"
        if plain_key not in state:
            continue  # weight missing from checkpoint too — genuine missing key
        w = state[plain_key].to(dtype=DTYPE, device=device)
        submod._parameters["weight"] = nn.Parameter(w, requires_grad=False)
        _restored.append(mod_name)
    if _restored:
        print(
            f"  Restored plain BF16 weights for {len(_restored)} module(s) "
            f"(stored as dense float in checkpoint, not packed NVFP4): "
            f"{_restored[:3]}{'…' if len(_restored) > 3 else ''}"
        )

    # ── Attach NVFP4 buffers directly from the checkpoint ────────────────────
    # The HF quantizer doesn't always create a slot for every NVFP4 buffer
    # (older/normalized configs frequently omit weight_global_scale), so
    # load_state_dict silently drops those keys into `unexpected`.  We forcibly
    # attach whatever the checkpoint actually provides, so _decompress_weight
    # always sees a complete, self-consistent {packed, scale, global_scale} set
    # regardless of what the quantizer did.
    _QUANT_SUFFIXES = (
        "weight_packed", "weight_scale", "weight_global_scale",
        "weight_shape", "weight_zero_point",
    )
    _mods = dict(module.named_modules())
    _attached = 0
    for key, tensor in state.items():
        for suf in _QUANT_SUFFIXES:
            if key.endswith("." + suf):
                submod = _mods.get(key[: -(len(suf) + 1)])
                if submod is not None and getattr(submod, suf, None) is None:
                    submod.register_buffer(suf, tensor.to(device), persistent=False)
                    _attached += 1
                break
    if _attached:
        print(f"  Attached {_attached} NVFP4 buffer(s) directly from checkpoint")

    # ── Sanity-check attention weights ───────────────────────────────────────
    # If q_proj.weight is still None after all loading attempts, the checkpoint
    # is missing the attention weights entirely (wrong format or wrong layer).
    try:
        attn = module.self_attn
        q_w = getattr(attn.q_proj, "weight", None)
        q_packed = getattr(attn.q_proj, "weight_packed", None)
        if q_w is None and q_packed is None:
            # Print what keys ARE present so the user can diagnose
            attn_keys = sorted(k for k in state if "attn" in k or "proj" in k)
            print(
                f"  WARNING: self_attn.q_proj has neither .weight nor "
                f".weight_packed after loading.\n"
                f"  Attention-related keys found in state dict "
                f"({len(attn_keys)}): {attn_keys[:8]}"
                f"{'…' if len(attn_keys) > 8 else ''}\n"
                f"  If this list is empty the checkpoint may use a different "
                f"key prefix — run the diagnostic below:\n"
                f"    python3 -c \""
                f"import json; wm=json.load(open('{model_dir}/model.safetensors.index.json'))['weight_map']; "
                f"[print(k) for k in sorted(wm) if 'layers.0' in k]\""
            )
    except AttributeError:
        pass

    module.eval()
    print(
        f"  Loaded {len(state)} tensors from {len(shards_needed)} shard(s) "
        f"({time.perf_counter() - t0:.1f}s total)"
    )
    return module


def _load_hf_model_full(
    model_dir: str,
    label: str,
    num_gpus: Optional[int] = None,
) -> nn.Module:
    """Load full NVFP4 checkpoint across GPUs via device_map='auto'."""
    from transformers import AutoConfig, AutoModelForCausalLM

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    n_gpu = num_gpus or torch.cuda.device_count()
    if n_gpu < 1:
        raise RuntimeError("--load-mode full requires at least one CUDA GPU")

    max_memory = _gpu_max_memory(n_gpu)
    t0 = time.perf_counter()
    print(f"\nLoading {label} (full model, {n_gpu} GPU(s)) from {model_dir} ...")
    print(f"  dtype: {DTYPE}  max_memory: {max_memory}")
    print("  NOTE: For Qwen3-480B this takes several minutes and ~240 GB RAM.")

    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        config=config,
        torch_dtype=DTYPE,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    elapsed = time.perf_counter() - t0
    _print_model_device_map(model, label)
    print(f"  Full model loaded in {elapsed / 60:.1f} min")
    return model


def _extract_layer_from_full_model(
    full_model: nn.Module,
    layer_path: str,
    device: str,
) -> nn.Module:
    """Clone the target layer to `device`, then free the full model."""
    layer = _get_nested_attr(full_model, layer_path)
    standalone = copy.deepcopy(layer)
    del full_model
    gc.collect()
    torch.cuda.empty_cache()
    return standalone.to(device=device, dtype=DTYPE).eval()


def print_model_keys(model_dir: str) -> None:
    """
    Print module paths from the safetensors index (no weight load).
    Use this to discover the correct --layer-path value for your checkpoint.
    """
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        raise FileNotFoundError(
            f"No index file at {index_path}. Cannot list keys without a full model load."
        )
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    module_paths = sorted({k.rsplit(".", 1)[0] for k in weight_map})
    print(f"\nModule paths from {index_path} ({len(module_paths)} total):")
    print("(Derived from tensor names — no weights loaded.)\n")

    prefixes: set = set()
    for path in module_paths:
        parts = path.split(".")
        for depth in (2, 3, 4, 5):
            if len(parts) >= depth:
                prefixes.add(".".join(parts[:depth]))
    print("Common prefixes (starting points for --layer-path):")
    for p in sorted(prefixes)[:30]:
        print(f"  {p}")
    print("\nSample leaf module paths (first 30):")
    for path in module_paths[:30]:
        print(f"  {path}")
    if len(module_paths) > 30:
        print(f"  ... ({len(module_paths) - 30} more)")
    print(
        "\nTIP: For Qwen3-MoE use --layer-path model.layers.0 (full decoder layer).\n"
        "     Adjust the index (0) to benchmark a different layer."
    )


def load_models(
    base_dir: str,
    layer_path: str,
    *,
    benchmark_mode: str = "runtime-patch",
    variant: str = "V3",
    load_mode: str = "layer",
    device: str = "cuda:0",
    num_gpus: Optional[int] = None,
) -> Tuple[nn.Module, nn.Module, int]:
    """
    Build (fused_bench, nonfused_bench, hidden_dim).

    runtime-patch (default)
        Loads the non-fused checkpoint once, builds both bench modules from the
        same layer weights. Cleanest kernel comparison.

    checkpoints
        Loads models/non-fused/ and models/fused/ separately. The fused bench
        applies the CUDA kernel to the offline-fused weights; the non-fused
        bench runs without any kernel patch.
    """
    nonfused_dir = os.path.join(base_dir, "models", "non-fused")
    fused_dir    = os.path.join(base_dir, "models", "fused")

    if benchmark_mode == "runtime-patch":
        print(f"  mode=runtime-patch  variant={variant}  load={load_mode}")
        if load_mode == "layer":
            layer = _load_layer_gpu(nonfused_dir, layer_path, "non-fused", device)
        else:
            full = _load_hf_model_full(nonfused_dir, "non-fused", num_gpus)
            layer = _extract_layer_from_full_model(full, layer_path, device)

        nonfused_bench = _build_unfused_bench(layer, device)
        fused_bench    = _build_fused_bench(layer, variant, device)
        hidden_dim     = layer.input_layernorm.weight.shape[0]
        del layer
        gc.collect()
        torch.cuda.empty_cache()

    elif benchmark_mode == "checkpoints":
        print(f"  mode=checkpoints  variant={variant}  load={load_mode}")
        if load_mode == "layer":
            layer_nf = _load_layer_gpu(nonfused_dir, layer_path, "non-fused", device)
            layer_f  = _load_layer_gpu(fused_dir,    layer_path, "fused",     device)
        else:
            full_nf  = _load_hf_model_full(nonfused_dir, "non-fused", num_gpus)
            layer_nf = _extract_layer_from_full_model(full_nf, layer_path, device)
            full_f   = _load_hf_model_full(fused_dir, "fused", num_gpus)
            layer_f  = _extract_layer_from_full_model(full_f, layer_path, device)

        nonfused_bench = _build_unfused_bench(layer_nf, device)
        fused_bench    = _build_fused_bench(layer_f, variant, device)
        hidden_dim     = layer_nf.input_layernorm.weight.shape[0]
        del layer_nf, layer_f
        gc.collect()
        torch.cuda.empty_cache()

    else:
        raise ValueError(
            f"Unknown --benchmark-mode {benchmark_mode!r}. Choose runtime-patch or checkpoints."
        )

    print(f"  Non-fused bench: {type(nonfused_bench).__name__}")
    print(f"  Fused bench:     {type(fused_bench).__name__}  (kernel variant {variant})")
    print(f"  Hidden dim:      {hidden_dim}")
    return fused_bench, nonfused_bench, hidden_dim


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

@dataclass
class ShapeResult:
    batch:   int
    seq_len: int
    hidden:  int
    out_dim: int

    fused_latencies:    List[float] = field(default_factory=list)
    nonfused_latencies: List[float] = field(default_factory=list)

    fused_peak_mem_mb:    float = 0.0
    nonfused_peak_mem_mb: float = 0.0

    max_abs_diff:  float = 0.0
    cosine_sim:    float = 0.0
    kl_divergence: float = 0.0

    @property
    def fused_median_ms(self) -> float:
        return statistics.median(self.fused_latencies) if self.fused_latencies else float("nan")

    @property
    def nonfused_median_ms(self) -> float:
        return statistics.median(self.nonfused_latencies) if self.nonfused_latencies else float("nan")

    @property
    def fused_p99_ms(self) -> float:
        return _percentile(self.fused_latencies, 99)

    @property
    def nonfused_p99_ms(self) -> float:
        return _percentile(self.nonfused_latencies, 99)

    @property
    def speedup(self) -> float:
        if self.fused_median_ms == 0:
            return float("nan")
        return self.nonfused_median_ms / self.fused_median_ms

    @property
    def fused_throughput(self) -> float:
        return (self.batch * self.seq_len) / (self.fused_median_ms / 1000.0)

    @property
    def nonfused_throughput(self) -> float:
        return (self.batch * self.seq_len) / (self.nonfused_median_ms / 1000.0)


def _percentile(data: List[float], pct: int) -> float:
    if not data:
        return float("nan")
    sd = sorted(data)
    return sd[min(int(len(sd) * pct / 100), len(sd) - 1)]


def _sync_time(fn, *args) -> float:
    start = time.perf_counter()
    fn(*args)
    torch.cuda.synchronize()
    return time.perf_counter() - start


def measure_latency(
    model: nn.Module,
    x: torch.Tensor,
    warmup: int = WARMUP_ITERS,
    measure: int = MEASURE_ITERS,
) -> List[float]:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize()
        latencies = []
        for _ in range(measure):
            latencies.append(_sync_time(model, x) * 1000.0)
    return latencies


def measure_peak_memory(model: nn.Module, x: torch.Tensor) -> float:
    model.eval()
    dev = x.device
    torch.cuda.reset_peak_memory_stats(dev)
    with torch.no_grad():
        model(x)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(dev) / 1024 ** 2


def measure_numerical_equivalence(
    fused: nn.Module,
    nonfused: nn.Module,
    x: torch.Tensor,
    n_iters: int = NUMERICAL_ITERS,
) -> Tuple[float, float, float]:
    """
    Compare fused vs non-fused outputs on the same input.
    Both modules return [batch, seq_len, q_dim + k_dim + v_dim].
    """
    fused.eval()
    nonfused.eval()
    max_diffs, cosines, kls = [], [], []
    with torch.no_grad():
        for _ in range(n_iters):
            out_f  = fused(x).float()
            out_nf = nonfused(x).float()

            max_diffs.append((out_f - out_nf).abs().max().item())

            f_flat  = out_f.view(out_f.size(0), -1)
            nf_flat = out_nf.view(out_nf.size(0), -1)
            cosines.append(F.cosine_similarity(f_flat, nf_flat, dim=1).mean().item())

            p  = F.softmax(out_f,  dim=-1).clamp(min=1e-10)
            q  = F.softmax(out_nf, dim=-1).clamp(min=1e-10)
            kls.append((p * (p / q).log()).sum(dim=-1).mean().item())

    return (
        statistics.mean(max_diffs),
        statistics.mean(cosines),
        statistics.mean(kls),
    )


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_benchmark(
    base_dir: str,
    layer_path: str,
    *,
    benchmark_mode: str = "runtime-patch",
    variant: str = "V3",
    load_mode: str = "layer",
    device: str = "cuda:0",
    num_gpus: Optional[int] = None,
) -> List[ShapeResult]:
    results: List[ShapeResult] = []

    fused_model, nonfused_model, hidden_dim = load_models(
        base_dir, layer_path,
        benchmark_mode=benchmark_mode,
        variant=variant,
        load_mode=load_mode,
        device=device,
        num_gpus=num_gpus,
    )

    def _ensure_device(m: nn.Module) -> nn.Module:
        try:
            if next(m.parameters()).device != torch.device(device):
                return m.to(device, dtype=DTYPE)
        except StopIteration:
            try:
                if next(m.buffers()).device != torch.device(device):
                    return m.to(device, dtype=DTYPE)
            except StopIteration:
                pass
        return m

    fused_model    = _ensure_device(fused_model).eval()
    nonfused_model = _ensure_device(nonfused_model).eval()

    shape_sweep = [
        (batch, seq_len, hidden_dim)
        for (batch, seq_len, _, _) in SHAPE_SWEEP
    ]

    for (batch, seq_len, hidden) in shape_sweep:
        print(f"\n{'='*60}")
        print(f"Shape: batch={batch}  seq={seq_len}  hidden={hidden}")
        print(f"{'='*60}")

        x = torch.randn(batch, seq_len, hidden, device=device, dtype=DTYPE)
        result = ShapeResult(batch=batch, seq_len=seq_len, hidden=hidden, out_dim=hidden)

        print("  Measuring latency (non-fused)...")
        result.nonfused_latencies = measure_latency(nonfused_model, x)

        print("  Measuring latency (fused)...")
        result.fused_latencies = measure_latency(fused_model, x)

        print("  Measuring peak memory...")
        result.nonfused_peak_mem_mb = measure_peak_memory(nonfused_model, x)
        result.fused_peak_mem_mb    = measure_peak_memory(fused_model, x)

        print("  Measuring numerical equivalence...")
        (result.max_abs_diff,
         result.cosine_sim,
         result.kl_divergence) = measure_numerical_equivalence(fused_model, nonfused_model, x)

        print(
            f"\n  Latency (median ms): fused={result.fused_median_ms:.3f}  "
            f"non-fused={result.nonfused_median_ms:.3f}  speedup={result.speedup:.2f}x"
        )
        print(
            f"  Latency (p99 ms):    fused={result.fused_p99_ms:.3f}  "
            f"non-fused={result.nonfused_p99_ms:.3f}"
        )
        print(
            f"  Throughput (tok/s):  fused={result.fused_throughput:,.0f}  "
            f"non-fused={result.nonfused_throughput:,.0f}"
        )
        print(
            f"  Peak mem (MB):       fused={result.fused_peak_mem_mb:.1f}  "
            f"non-fused={result.nonfused_peak_mem_mb:.1f}"
        )
        print(f"  Numerical equivalence:")
        print(f"    max |diff|  = {result.max_abs_diff:.6f}")
        print(f"    cosine sim  = {result.cosine_sim:.6f}  (1.0 = identical)")
        print(f"    KL div      = {result.kl_divergence:.6f}  (0.0 = identical)")

        del x
        gc.collect()
        torch.cuda.empty_cache()
        results.append(result)

    return results


def print_summary_table(results: List[ShapeResult]) -> None:
    header = (
        f"{'batch':>5} {'seq':>5} {'hidden':>7} "
        f"{'fused_med':>10} {'nf_med':>10} {'speedup':>8} "
        f"{'fused_p99':>10} {'nf_p99':>10} "
        f"{'fused_mem':>10} {'nf_mem':>10} "
        f"{'cos_sim':>9} {'kl_div':>9}"
    )
    print(f"\n{'='*len(header)}")
    print("SUMMARY TABLE")
    print(f"{'='*len(header)}")
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.batch:>5} {r.seq_len:>5} {r.hidden:>7} "
            f"{r.fused_median_ms:>10.3f} {r.nonfused_median_ms:>10.3f} {r.speedup:>8.2f}x "
            f"{r.fused_p99_ms:>10.3f} {r.nonfused_p99_ms:>10.3f} "
            f"{r.fused_peak_mem_mb:>10.1f} {r.nonfused_peak_mem_mb:>10.1f} "
            f"{r.cosine_sim:>9.6f} {r.kl_divergence:>9.6f}"
        )
    print(f"{'='*len(header)}")


def _default_results_dir() -> str:
    return os.path.join(_BENCH_DIR, "results")


def _make_results_path(
    output_dir: str,
    *,
    benchmark_mode: str,
    variant: str,
    load_mode: str,
) -> str:
    ts   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = f"benchmark_{ts}_{benchmark_mode}_{variant}_{load_mode}.csv"
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, base)
    if os.path.exists(path):
        raise FileExistsError(f"Refusing to overwrite: {path}")
    return path


def save_csv(
    results: List[ShapeResult],
    path: str,
    *,
    run_metadata: Optional[dict] = None,
) -> str:
    import csv

    meta = run_metadata or {}
    fields = [
        "run_timestamp_utc", "benchmark_mode", "variant", "load_mode", "device",
        "batch", "seq_len", "hidden", "out_dim",
        "fused_median_ms", "nonfused_median_ms", "speedup",
        "fused_p99_ms", "nonfused_p99_ms",
        "fused_throughput", "nonfused_throughput",
        "fused_peak_mem_mb", "nonfused_peak_mem_mb",
        "max_abs_diff", "cosine_sim", "kl_divergence",
    ]
    with open(path, "x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({
                **meta,
                "batch": r.batch, "seq_len": r.seq_len,
                "hidden": r.hidden, "out_dim": r.out_dim,
                "fused_median_ms":    r.fused_median_ms,
                "nonfused_median_ms": r.nonfused_median_ms,
                "speedup":            r.speedup,
                "fused_p99_ms":       r.fused_p99_ms,
                "nonfused_p99_ms":    r.nonfused_p99_ms,
                "fused_throughput":   r.fused_throughput,
                "nonfused_throughput": r.nonfused_throughput,
                "fused_peak_mem_mb":    r.fused_peak_mem_mb,
                "nonfused_peak_mem_mb": r.nonfused_peak_mem_mb,
                "max_abs_diff":  r.max_abs_diff,
                "cosine_sim":    r.cosine_sim,
                "kl_divergence": r.kl_divergence,
            })
    print(f"\nResults saved to: {path}")
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark RMSNorm+QKV fusion vs non-fused (Qwen3 / Qwen3-MoE NVFP4).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dir", required=True,
        help="Base directory containing models/non-fused/ (and models/fused/ for checkpoints mode).",
    )
    parser.add_argument(
        "--layer-path", default="model.layers.0",
        help=(
            "Dot-separated path to the Qwen3 decoder layer to benchmark "
            "(default: model.layers.0). Use --print-keys to discover the path."
        ),
    )
    parser.add_argument(
        "--benchmark-mode", choices=("runtime-patch", "checkpoints"), default="runtime-patch",
        help=(
            "runtime-patch (default): build fused+nonfused from the same non-fused weights — "
            "cleanest kernel comparison. "
            "checkpoints: load models/non-fused/ and models/fused/ separately."
        ),
    )
    parser.add_argument(
        "--variant", choices=("V1", "V3"), default="V3",
        help="Fused CUDA kernel variant: V1 (256 threads) or V3 (512 threads, default).",
    )
    parser.add_argument(
        "--load-mode", choices=("layer", "full"), default="layer",
        help=(
            "layer (default): load only --layer-path weights from shard files (~seconds). "
            "full: load entire model via device_map=auto, then clone the target layer."
        ),
    )
    parser.add_argument(
        "--device", default="cuda:0",
        help="CUDA device for benchmarking (default: cuda:0).",
    )
    parser.add_argument(
        "--num-gpus", type=int, default=None,
        help="Number of GPUs for --load-mode full (default: all visible).",
    )
    parser.add_argument(
        "--print-keys", action="store_true",
        help="Print module paths from the checkpoint index and exit (no weights loaded).",
    )
    parser.add_argument(
        "--test-load", action="store_true",
        help=(
            "Smoke test: load weights and run one forward pass per bench module, then exit. "
            "Does NOT run the full latency/memory/equivalence sweep."
        ),
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory for CSV output (default: benchmarks/results/).",
    )
    args = parser.parse_args()

    if args.print_keys:
        nonfused_dir = os.path.join(args.dir, "models", "non-fused")
        print_model_keys(nonfused_dir)
        sys.exit(0)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Run on a GPU machine.")

    n_gpu = torch.cuda.device_count()
    print(f"GPU ({n_gpu}x): {torch.cuda.get_device_name(0)}")
    print(f"PyTorch:        {torch.__version__}")
    print(f"CUDA:           {torch.version.cuda}")
    print(f"Base dir:       {args.dir}")
    print(f"Layer path:     {args.layer_path}")
    print(f"Benchmark mode: {args.benchmark_mode}")
    print(f"Variant:        {args.variant}")
    print(f"Load mode:      {args.load_mode}  |  Device: {args.device}")
    if args.benchmark_mode == "runtime-patch":
        print(f"Warmup: {WARMUP_ITERS}  Measure: {MEASURE_ITERS}  Numerical: {NUMERICAL_ITERS}")

    if args.test_load:
        fused, nonfused, hidden = load_models(
            args.dir, args.layer_path,
            benchmark_mode=args.benchmark_mode,
            variant=args.variant,
            load_mode=args.load_mode,
            device=args.device,
            num_gpus=args.num_gpus,
        )
        x = torch.randn(1, 128, hidden, device=args.device, dtype=DTYPE)
        with torch.no_grad():
            y_f  = fused(x)
            y_nf = nonfused(x)
        print(
            f"\nForward OK — fused={tuple(y_f.shape)}  non-fused={tuple(y_nf.shape)}"
        )
        used = [
            torch.cuda.memory_allocated(i) / 1024 ** 3
            for i in range(torch.cuda.device_count())
        ]
        print(f"GPU memory allocated (GiB): {[round(u, 2) for u in used]}")
        print(
            "\n--test-load passed. Re-run WITHOUT --test-load to run the full benchmark."
        )
        sys.exit(0)

    results = run_benchmark(
        args.dir, args.layer_path,
        benchmark_mode=args.benchmark_mode,
        variant=args.variant,
        load_mode=args.load_mode,
        device=args.device,
        num_gpus=args.num_gpus,
    )
    print_summary_table(results)

    output_dir = args.output_dir or _default_results_dir()
    out_path = _make_results_path(
        output_dir,
        benchmark_mode=args.benchmark_mode,
        variant=args.variant,
        load_mode=args.load_mode,
    )
    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    save_csv(
        results,
        out_path,
        run_metadata={
            "run_timestamp_utc": run_ts,
            "benchmark_mode":    args.benchmark_mode,
            "variant":           args.variant,
            "load_mode":         args.load_mode,
            "device":            args.device,
        },
    )
