"""
Pre-compute fused LayerNorm+Linear weights.

For LayerNorm(gamma, beta) followed by Linear(W, b):
    output = Linear(LayerNorm(x))
           = x @ (I - E/h) @ diag(gamma) @ W.T / std(x) + beta @ W.T + b

where std(x) = sqrt(v(x)^2/h + eps), v(x) = ||x - mean(x)||_2.

The forward pass computes: x @ W_new.T / std(x) + b_new
where std(x) is derived from v(x) returned by the CUDA kernel.

Notation: E = 11^T is the h x h all-ones matrix (outer product of the all-ones
vector 1 with itself). Thus (I - E/h) is the centering matrix that subtracts
the mean from each column.

Assumptions: LayerNorm has both weight (gamma) and bias (beta);
normalized_shape is 1-dimensional.

So:
    M = (W * gamma).T            # element-wise (W * gamma), then .T = diag(gamma) @ W.T, shape [h, out]
    F_new = M - M.mean(dim=0)    # = (I - E/h) @ M, center each column, shape [h, out]
    W_new = F_new.T              # shape [out, h]
    b_new = beta @ W.T + b       # matrix multiply beta with W.T + bias, shape [out]
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# NVFP4 / compressed-tensors helpers
# ---------------------------------------------------------------------------

# E2M1 (FP4) magnitude lookup: the 3 magnitude bits index these values;
# the 4th bit is the sign.  Fixed by the NVFP4 spec.
_E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)


def _dequantize_nvfp4(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Reconstruct a dense [out, in] weight from NVFP4 compressed-tensors buffers.

        weight_packed        uint8           (out, in // 2)   2 FP4 codes/byte
        weight_scale         float8_e4m3fn   (out, in // 16)  per-block scale
        weight_global_scale  float32         scalar           per-tensor scale

    Per element:  W = E2M1(code) * (weight_scale / weight_global_scale)

    The global scale stores the quantization scale, so we divide by it.  Block
    size is inferred from shapes (in // n_scale_cols == 16 for NVFP4).  This is
    pure PyTorch and independent of the installed compressed-tensors version.
    """
    device = weight_packed.device
    lut = _E2M1_LUT.to(device)

    packed = weight_packed.view(torch.uint8)
    out_dim, in_half = packed.shape

    low_nib = packed & 0x0F
    high_nib = (packed >> 4) & 0x0F

    def _decode(nib: torch.Tensor) -> torch.Tensor:
        sign = 1.0 - 2.0 * ((nib >> 3) & 1).to(torch.float32)
        return sign * lut[(nib & 0x07).long()]

    vals = torch.empty(out_dim, in_half * 2, device=device, dtype=torch.float32)
    vals[:, 0::2] = _decode(low_nib)
    vals[:, 1::2] = _decode(high_nib)

    in_dim = in_half * 2
    n_blocks = weight_scale.shape[1]
    block = in_dim // n_blocks

    scale = weight_scale.to(torch.float32) / weight_global_scale.to(torch.float32)
    scale = scale.repeat_interleave(block, dim=1)

    return (vals * scale).to(out_dtype)


def _decompress_weight(linear: nn.Module) -> torch.Tensor:
    """
    Return a dense [out, in] weight tensor for fusion math.

    Two cases, both read directly from the module's tensors (no dependency on
    any compressed-tensors decompression API):

      1. Dense .weight present  → return it.
      2. NVFP4 packed buffers   → dequantize with _dequantize_nvfp4().
    """
    w = getattr(linear, "weight", None)
    if w is not None:
        return w.data

    wp = getattr(linear, "weight_packed", None)
    if wp is not None:
        return _dequantize_nvfp4(
            wp, linear.weight_scale, linear.weight_global_scale
        )

    raise AttributeError(
        f"{type(linear).__name__} has neither a dense .weight nor NVFP4 "
        ".weight_packed buffers — the checkpoint did not load correctly."
    )


def compute_fused_weights(
    ln: nn.LayerNorm,
    linear: nn.Linear,
) -> tuple[torch.Tensor, torch.Tensor, int, float]:
    """
    Compute fused weights W_new, b_new for LayerNorm + Linear fusion.

    Args:
        ln: LayerNorm module with weight (gamma) and bias (beta)
        linear: Linear module with weight W [out, h] and optional bias b [out]

    Returns:
        W_new: [out, h] fused weight
        b_new: [out] fused bias
        h: hidden dimension (for std computation)
        eps: LayerNorm epsilon
    """
    gamma = ln.weight.data       # [h]
    beta = ln.bias.data          # [h]
    W = linear.weight.data       # [out, h]
    b = linear.bias.data if linear.bias is not None else torch.zeros(
        W.size(0), device=W.device, dtype=W.dtype
    )
    h = ln.normalized_shape[0]
    eps = ln.eps

    # M = diag(gamma) @ W.T = (W * gamma).T, shape [h, out]
    M = (W * gamma).T

    # (I - E/h) @ M: center each column (subtract column mean)
    F_new = M - M.mean(dim=0, keepdim=True)

    W_new = F_new.T              # [out, h]
    b_new = beta @ W.T + b      # [out]: beta @ W.T is like linear(beta)

    return W_new, b_new, h, eps


def compute_fused_weights_rmsnorm(
    rms_norm,
    linear: nn.Linear,
) -> tuple[torch.Tensor, torch.Tensor, int, float]:
    """
    Compute fused weights W_new, b_new for RMSNorm + Linear fusion.

    RMSNorm: output = x / rms(x) * gamma, where rms(x) = sqrt(mean(x^2) + eps).
    Unlike LayerNorm, there is no mean subtraction and typically no bias.

    The fused forward computes: x @ W_new.T / rms(x) + b_new
    where:
        W_new = W * gamma   (element-wise, no centering needed)
        b_new = b           (no beta term since RMSNorm has no bias)

    Args:
        rms_norm: RMSNorm module with weight (gamma), no bias
        linear: Linear module with weight W [out, h] and optional bias b [out]

    Returns:
        W_new: [out, h] fused weight
        b_new: [out] fused bias
        h: hidden dimension
        eps: RMSNorm epsilon
    """
    gamma = rms_norm.weight.data    # [h]
    W = _decompress_weight(linear)  # [out, h]; NVFP4-safe
    b = linear.bias.data if linear.bias is not None else torch.zeros(
        W.size(0), device=W.device, dtype=W.dtype
    )
    h = gamma.shape[0]
    eps = rms_norm.eps if hasattr(rms_norm, 'eps') else rms_norm.variance_epsilon

    gamma_cast = gamma.to(W.dtype)
    W_new = W * gamma_cast

    # b_new = b (no beta in RMSNorm)
    b_new = b

    return W_new, b_new, h, eps


def transform_opt_layer(decoder_layer) -> dict:
    """
    Compute fused weights for all LayerNorm+Linear pairs in an OPT decoder layer.

    Pairs:
        1. self_attn_layer_norm -> q_proj, k_proj, v_proj
        2. final_layer_norm -> fc1

    Returns:
        dict mapping projection name to (W_new, b_new)
    """
    attn = decoder_layer.self_attn
    ln1 = decoder_layer.self_attn_layer_norm
    ln2 = decoder_layer.final_layer_norm

    fused = {}

    # Attention projections share the same layer norm
    for name in ["q_proj", "k_proj", "v_proj"]:
        proj = getattr(attn, name)
        W_new, b_new, h, eps = compute_fused_weights(ln1, proj)
        fused[f"attn_{name}"] = (W_new, b_new, h, eps)

    # FFN fc1
    W_new, b_new, h, eps = compute_fused_weights(ln2, decoder_layer.fc1)
    fused["fc1"] = (W_new, b_new, h, eps)

    return fused


def compute_fused_weights_rmsnorm_combined(
    rms_norm,
    linears: list[nn.Linear],
) -> tuple[torch.Tensor, torch.Tensor, list[int], int, float]:
    """
    Compute fused weights for RMSNorm + multiple Linear layers that share the same norm.

    Concatenates the weight matrices along dim=0 so a single matmul replaces
    multiple separate projections (e.g. Q/K/V or gate/up).

    Args:
        rms_norm: RMSNorm module with weight (gamma)
        linears: list of Linear modules sharing this norm

    Returns:
        W_combined: [sum(out_dims), h] fused weight
        b_combined: [sum(out_dims)] fused bias
        split_sizes: list of output dims per linear, for torch.split
        h: hidden dimension
        eps: RMSNorm epsilon
    """
    gamma = rms_norm.weight.data    # [h]
    h = gamma.shape[0]
    eps = rms_norm.eps if hasattr(rms_norm, 'eps') else rms_norm.variance_epsilon

    W_parts = []
    b_parts = []
    split_sizes = []

    for linear in linears:
        W = _decompress_weight(linear)  # [out_i, h]; NVFP4-safe
        b = linear.bias.data if linear.bias is not None else torch.zeros(
            W.size(0), device=W.device, dtype=W.dtype
        )
        gamma_cast = gamma.to(W.dtype)
        W_parts.append(W * gamma_cast)  # element-wise, no centering
        b_parts.append(b)
        split_sizes.append(W.size(0))

    W_combined = torch.cat(W_parts, dim=0)  # [sum(out_dims), h]
    b_combined = torch.cat(b_parts, dim=0)  # [sum(out_dims)]

    return W_combined, b_combined, split_sizes, h, eps



# ---------------------------------------------------------------------------
# Qwen3-MoE
# ---------------------------------------------------------------------------

def transform_qwen3_layer(decoder_layer) -> dict:
    """
    Compute fused combined weights for a Qwen3 MoE decoder layer.

    Only fuses attention QKV:
        input_layernorm -> [self_attn.q_proj, self_attn.k_proj, self_attn.v_proj]

    The MoE MLP is NOT fused.  Qwen3-MoE routes tokens through a sparse set of
    expert FFNs after the post_attention_layernorm.  Because the router sits
    between the norm output and the experts, there is no single linear layer to
    absorb the norm into — the same reasoning that applies to GPT-OSS.

    Qwen3 also applies per-head QK norms (self_attn.q_norm / self_attn.k_norm)
    AFTER the projections.  These operate on the projected output, not on the
    residual stream, so they are independent of this fusion and are left
    untouched.

    Args:
        decoder_layer: a Qwen3MoeDecoderLayer instance

    Returns:
        dict with key "attn_qkv" mapping to
        (W_combined, b_combined, split_sizes, h, eps)
    """
    attn = decoder_layer.self_attn
    norm = decoder_layer.input_layernorm

    qkv_linears = [attn.q_proj, attn.k_proj, attn.v_proj]
    return {"attn_qkv": compute_fused_weights_rmsnorm_combined(norm, qkv_linears)}
