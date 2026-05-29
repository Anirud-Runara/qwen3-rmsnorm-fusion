"""
Monkey-patch a Qwen3 / Qwen3-MoE model to use fused RMSNorm+Linear modules.

Replaces the forward pass of each Qwen3 decoder layer so that:
  - input_layernorm + q/k/v_proj -> fused_qkv (combined: one matmul + one
    RMSNorm-normalize kernel call)

Only attention QKV is fused.  The MLP (dense or sparse MoE) is left untouched:
for MoE the router dispatches tokens between the norm output and the experts,
making norm-into-expert fusion impractical.

Design notes (transformers >= 5.x):
  - The attention forward now returns (attn_output, attn_weights) and dispatches
    through ALL_ATTENTION_FUNCTIONS (which applies causal masking, sliding
    window, and the configured SDPA/flash/eager kernel).  We reproduce that
    exact post-projection pipeline rather than re-implementing attention by
    hand, and we pull the helper symbols (apply_rotary_pos_emb,
    eager_attention_forward, ALL_ATTENTION_FUNCTIONS) from the attention
    class's own module so the same patch works for dense Qwen3 and Qwen3-MoE.
  - The decoder layer forward returns a bare hidden_states tensor.
  - Per-head QK norms (q_norm / k_norm) run AFTER the projection on the
    projected Q/K, exactly as upstream — independent of this fusion.
  - input_layernorm is SKIPPED in the layer forward; its scale (gamma) and the
    1/rms normalization are absorbed into the fused QKV module.

Usage:
    from src.patch_qwen3 import patch_qwen3_model
    model = AutoModelForCausalLM.from_pretrained(...)
    model = patch_qwen3_model(model, device="cuda", variant="V3")
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


def patch_qwen3_model(model, device=None, variant="V3"):
    """
    Patch all decoder layers in a Qwen3 / Qwen3-MoE model to use fused RMSNorm+QKV.

    Args:
        model: HuggingFace Qwen3ForCausalLM or Qwen3MoeForCausalLM (BF16, CPU or GPU)
        device: target device for fused weight tensors (defaults to model device)
        variant: kernel variant -- "V1" (256 threads) or "V3" (512 threads, recommended)

    Returns:
        The patched model (modified in-place)
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
    """Patch a single Qwen3 decoder layer with fused RMSNorm+QKV."""
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
    Replace attention forward to use fused QKV (skipping the standalone
    input_layernorm + q/k/v_proj), then reproduce the upstream post-projection
    pipeline verbatim (q_norm/k_norm, RoPE, cache, attention dispatch, o_proj).
    """
    # Pull the exact helper symbols the real forward uses, from the attention
    # class's own module (modeling_qwen3 or modeling_qwen3_moe).
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)

        # ------------------------------------------------------------------ #
        # Fused RMSNorm + Q/K/V projections in one call.                      #
        # input_layernorm is baked into the fused weights; skip it here.      #
        # q_raw/k_raw/v_raw == q_proj(input_layernorm(x)) etc.                #
        # ------------------------------------------------------------------ #
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
    post_attention_layernorm and the MLP (dense or MoE) run unchanged.
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

        # input_layernorm is SKIPPED — it is fused into fused_qkv weights.
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

        # MLP path: post_attention_layernorm runs as normal. The dense Qwen3
        # mlp returns a tensor; the sparse Qwen3-MoE mlp returns (hidden, router).
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        mlp_out = layer.mlp(hidden_states)
        if isinstance(mlp_out, tuple):
            mlp_out = mlp_out[0]
        hidden_states = residual + mlp_out

        return hidden_states

    layer.forward = patched_forward
