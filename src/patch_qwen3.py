"""
Monkey-patch a Qwen3-MoE model to use fused RMSNorm+Linear modules.

Replaces the forward pass of each Qwen3 MoE decoder layer so that:
  - input_layernorm + q/k/v_proj -> fused_qkv (combined)

Only attention QKV is fused.  The sparse MoE MLP is left untouched because
the router dispatches tokens between the norm output and the experts, making
norm-into-expert fusion impractical — same reasoning as GPT-OSS.

Qwen3-specific notes:
  - Per-head QK norms (self_attn.q_norm / self_attn.k_norm) are applied AFTER
    the projections.  They operate on projected Q/K tensors, not on the
    residual stream, so they are independent of this fusion and run unchanged.
  - Qwen3 uses standard (non-interleaved) RoPE.
  - Projections do NOT have bias (bias=False in the config).

Usage:
    from src.patch_qwen3 import patch_qwen3_model
    model = AutoModelForCausalLM.from_pretrained(...)
    model = patch_qwen3_model(model, device="cuda", variant="V3")
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeDecoderLayer,
    Qwen3MoeAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)
import torch.nn.functional as F

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
    Patch all decoder layers in a Qwen3-MoE model to use fused RMSNorm+QKV.

    Only attention QKV is fused (combined mode). MoE MLP is untouched.

    Args:
        model: HuggingFace Qwen3MoeForCausalLM model (BF16, on CPU or GPU)
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

    for layer_idx, layer in enumerate(model.model.layers):
        _patch_decoder_layer(layer, device, variant)

    return model


def _patch_decoder_layer(layer: Qwen3MoeDecoderLayer, device, variant: str = "V3"):
    """Patch a single Qwen3 MoE decoder layer with fused RMSNorm+QKV."""
    fused_weights = transform_qwen3_layer(layer)
    cls = _COMBINED_VARIANT_CLASSES[variant]

    W_comb, b_comb, split_sizes, h, eps = fused_weights["attn_qkv"]
    layer.self_attn.fused_qkv = cls(
        W_comb.to(device), b_comb.to(device), split_sizes, h, eps
    )

    _patch_attention_forward(layer.self_attn)
    _patch_layer_forward(layer)


def _patch_attention_forward(attn: Qwen3MoeAttention):
    """
    Replace attention forward to use fused QKV (skipping standalone RMSNorm call).

    The per-head q_norm and k_norm are preserved — they run on the projected
    Q and K tensors after splitting, exactly as in the original forward.
    """

    def patched_forward(
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # ------------------------------------------------------------------ #
        # Fused RMSNorm + Q/K/V projections in one call                       #
        # input_layernorm is baked into the fused weights; skip it here.      #
        # ------------------------------------------------------------------ #
        q_raw, k_raw, v_raw = attn.fused_qkv(hidden_states)

        # Reshape: [bsz, seq, head_dim * n_heads] -> [bsz, n_heads, seq, head_dim]
        query_states = q_raw.view(bsz, q_len, attn.num_heads, attn.head_dim).transpose(1, 2)
        key_states   = k_raw.view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
        value_states = v_raw.view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)

        # Per-head QK norms (Qwen3-specific) — preserved, run after projection
        query_states = attn.q_norm(query_states)
        key_states   = attn.k_norm(key_states)

        # RoPE
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # KV cache update
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, attn.layer_idx, cache_kwargs
            )

        # GQA: expand KV heads to match Q heads
        key_states   = repeat_kv(key_states,   attn.num_key_value_groups)
        value_states = repeat_kv(value_states, attn.num_key_value_groups)

        # Scaled dot-product attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * attn.scaling

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=attn.attention_dropout, training=attn.training)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, attn.num_heads * attn.head_dim)
        attn_output = attn.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    attn.forward = patched_forward


def _patch_layer_forward(layer: Qwen3MoeDecoderLayer):
    """
    Replace decoder layer forward to skip input_layernorm (fused into QKV).

    post_attention_layernorm and the sparse MoE MLP run completely unchanged.
    """

    def patched_forward(
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple:
        residual = hidden_states

        # input_layernorm is SKIPPED here — it is fused into fused_qkv weights.
        # The attention module receives the raw residual stream directly.
        hidden_states, self_attn_weights, present_key_value = layer.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # MoE MLP path: post_attention_layernorm runs as normal
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states, router_logits = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs

    layer.forward = patched_forward
