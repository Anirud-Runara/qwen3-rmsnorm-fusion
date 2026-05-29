"""
Monkey-patch a Qwen3 / Qwen3-MoE model to use fused RMSNorm+Linear modules.

Replaces the forward pass of each Qwen3 decoder layer so that:
  - input_layernorm + q/k/v_proj -> fused_qkv (combined: one matmul + one
    CUDA normalize kernel) at RUNTIME, for maximum throughput.

IMPORTANT — two-phase approach
-------------------------------
1. Offline weight transform (Test A / fuse_model.py):
   `_offline_fuse_layer` simply modifies weights in-place (W *= gamma,
   gamma = 1) and lets the original HF forward run. No CUDA extension needed.
   This is the artifact that gets stored and quantized. It does NOT speed
   anything up on its own.

2. Runtime CUDA kernel (Test B / serving + benchmark — THIS file):
   `patch_qwen3_model` swaps in the fused CUDA kernel. This is where the
   speedup comes from. Requires the compiled extension (JIT via load_cuda).

The runtime forward delegates attention to transformers' own
ALL_ATTENTION_FUNCTIONS dispatch, so the attention backend is IDENTICAL to the
unpatched baseline. That is required for a fair benchmark: the measured delta
is then attributable purely to the RMSNorm+QKV fusion, not to an attention-impl
difference.

Architecture notes (Qwen3 / Qwen3-MoE)
---------------------------------------
- input_layernorm -> [q_proj, k_proj, v_proj]   <- FUSED HERE
- Per-head q_norm / k_norm run AFTER projection; independent of this fusion.
- MoE MLP NOT fused (router sits between norm and experts).
- Dense MLP NOT fused either (out of scope).

Tested with transformers >= 5.0 (uses the ALL_ATTENTION_FUNCTIONS dispatch and
the bare-tensor decoder-layer return convention).
"""

import sys
import torch
from typing import Optional, Tuple

from src.weight_transform import transform_qwen3_layer
from src.fused_forward import (
    FusedRMSNormCombinedLinearV1,
    FusedRMSNormCombinedLinearV3,
)

_COMBINED_VARIANT_CLASSES = {
    "V1": FusedRMSNormCombinedLinearV1,
    "V3": FusedRMSNormCombinedLinearV3,
}


# ------------------------------------------------------------------ #
# Offline helper (Test A / fuse_model.py path — no CUDA kernel)        #
# ------------------------------------------------------------------ #

def _offline_fuse_layer(layer) -> None:
    """
    In-place offline fusion: W *= gamma, gamma = 1.
    Uses the original HF forward — no CUDA extension needed.
    This is what fuse_model.py does to the 480B checkpoint.
    """
    attn = layer.self_attn
    gamma = layer.input_layernorm.weight.data.clone()
    for proj in (attn.q_proj, attn.k_proj, attn.v_proj):
        proj.weight.data.mul_(gamma)
    layer.input_layernorm.weight.data.fill_(1.0)


# ------------------------------------------------------------------ #
# Runtime CUDA patch (Test B / serving)                                #
# ------------------------------------------------------------------ #

def patch_qwen3_model(model, device=None, variant="V3"):
    """
    Patch all decoder layers to use the fused RMSNorm+QKV CUDA kernel.

    Args:
        model: Qwen3ForCausalLM or Qwen3MoeForCausalLM
        device: target device (default: model's current device)
        variant: "V1" (256 threads) or "V3" (512 threads)
    """
    if variant not in _COMBINED_VARIANT_CLASSES:
        raise ValueError(
            f"Unknown variant {variant!r}; choose from {list(_COMBINED_VARIANT_CLASSES)}"
        )
    if device is None:
        device = next(model.parameters()).device

    for layer in model.model.layers:
        _patch_decoder_layer(layer, device, variant)

    return model


def _patch_decoder_layer(layer, device, variant: str = "V3"):
    """Patch a single decoder layer."""
    fused_weights = transform_qwen3_layer(layer)
    cls = _COMBINED_VARIANT_CLASSES[variant]

    W_comb, b_comb, split_sizes, h, eps = fused_weights["attn_qkv"]
    layer.self_attn.fused_qkv = cls(
        W_comb.to(device), b_comb.to(device), split_sizes, h, eps
    )

    _patch_attention_forward(layer.self_attn)
    _patch_layer_forward(layer)


def _patch_attention_forward(attn):
    """
    Replace attention forward to use fused QKV (skipping standalone RMSNorm),
    then reproduce the upstream post-projection pipeline verbatim.

    Helper symbols are pulled from the attention class's own module so the same
    patch works for dense Qwen3 (modeling_qwen3) and Qwen3-MoE
    (modeling_qwen3_moe).  Attention itself is dispatched through
    ALL_ATTENTION_FUNCTIONS — identical backend to the unpatched model.
    """
    _mod = sys.modules[type(attn).__module__]
    apply_rotary_pos_emb = _mod.apply_rotary_pos_emb
    eager_attention_forward = _mod.eager_attention_forward
    ALL_ATTENTION_FUNCTIONS = _mod.ALL_ATTENTION_FUNCTIONS

    def patched_forward(
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)

        # Fused RMSNorm + Q/K/V projections (input_layernorm folded into weights)
        q_raw, k_raw, v_raw = attn.fused_qkv(hidden_states)

        # Per-head QK norms run on the projected Q/K, exactly as upstream.
        query_states = attn.q_norm(q_raw.view(hidden_shape)).transpose(1, 2)
        key_states   = attn.k_norm(k_raw.view(hidden_shape)).transpose(1, 2)
        value_states = v_raw.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, attn.layer_idx
            )

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            attn.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            attn,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not attn.training else attn.attention_dropout,
            scaling=attn.scaling,
            sliding_window=attn.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn.o_proj(attn_output)
        return attn_output, attn_weights

    attn.forward = patched_forward


def _patch_layer_forward(layer):
    """
    Replace decoder layer forward to skip input_layernorm (fused into QKV).
    Returns a bare hidden_states tensor (transformers >= 5.x convention).
    Handles dense (tensor) and MoE ((hidden, router_logits)) MLP returns.
    """

    def patched_forward(
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        use_cache: Optional[bool] = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states

        # input_layernorm is SKIPPED — fused into fused_qkv weights.
        hidden_states, _ = layer.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # MLP path — post_attention_layernorm runs as normal.
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        mlp_out = layer.mlp(hidden_states)
        if isinstance(mlp_out, (tuple, list)):
            mlp_out = mlp_out[0]
        hidden_states = residual + mlp_out

        return hidden_states

    layer.forward = patched_forward
