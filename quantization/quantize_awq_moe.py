#!/usr/bin/env python3
"""
AWQ W4A16 quantization for Qwen3-MoE with llm-compressor — fusion-compatible.

Validate on Qwen3-30B-A3B first, then scale to the fused 480B.

WHAT THIS DOES DIFFERENTLY FROM A STANDARD AWQ (e.g. QuantTrio's):
  QuantTrio quantizes everything except the MoE router gate (.mlp.gate),
  INCLUDING attention q/k/v. Our RMSNorm-fusion V1 kernel does a bf16 matmul on
  the combined QKV, so it cannot consume INT4 q/k/v. Therefore we KEEP
  q_proj/k_proj/v_proj in bf16 (ignored) — this also protects input_layernorm
  (which the fusion set to gamma=1) from AWQ's scale absorption. Everything else
  (MoE experts, o_proj, ...) is quantized to 4-bit. .mlp.gate stays bf16, like
  QuantTrio.

ENV WARNING:
  llm-compressor may not yet support transformers 5.x. Run this in a DEDICATED
  venv and let pip pin a compatible transformers (do NOT force 5.9). See
  quantization/README.md. The produced checkpoint is just files and can be
  served later in a transformers-5.x env with patch_qwen3_model(variant="V1").

Usage (validation on the small MoE first):
    python3 quantize_awq_moe.py \\
        --model-id Qwen/Qwen3-30B-A3B \\
        --output-dir ./qwen3-30b-a3b-awq

Then the fused 480B (point --model-id at your fused checkpoint dir):
    python3 quantize_awq_moe.py \\
        --model-id /workspace/qwen3-480b-fused-bf16 \\
        --output-dir /workspace/qwen3-480b-fused-awq \\
        --device-map auto
"""

import argparse

# AWQModifier import path has churned across llm-compressor versions — try the
# known locations (0.9 generalized -> .awq; 0.8 -> .transform.awq; some -> .quantization).
try:
    from llmcompressor.modifiers.awq import AWQModifier
except ImportError:
    try:
        from llmcompressor.modifiers.transform.awq import AWQModifier
    except ImportError:
        from llmcompressor.modifiers.quantization import AWQModifier

from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor import oneshot
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def parse_args():
    p = argparse.ArgumentParser(description="Fusion-compatible AWQ W4A16 for Qwen3-MoE")
    p.add_argument("--model-id", default="Qwen/Qwen3-30B-A3B",
                   help="HF id or local path (use your fused 480B dir for the real run)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--calib-dataset", default="ise-uiuc/Magicoder-Evol-Instruct-110K",
                   help="Code-instruction dataset; calibrate on data like the inference workload")
    p.add_argument("--calib-samples", type=int, default=512)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--device-map", default="auto",
                   help='"auto" fits the 30B on 1 GPU; for the 480B use offload / multi-GPU')
    p.add_argument("--quantize-qkv", action="store_true",
                   help="Quantize attention q/k/v too (breaks the V1 fusion kernel). "
                        "Default keeps them bf16 for fusion compatibility.")
    return p.parse_args()


def build_calibration(tokenizer, args):
    """Code data formatted with the chat template (instruct model -> match
    inference distribution). Adjust field names per dataset if you swap it."""
    ds = load_dataset(args.calib_dataset, split="train")
    ds = ds.shuffle(seed=42).select(range(args.calib_samples))

    def to_messages(ex):
        instr = ex.get("instruction") or ex.get("problem") or ex.get("prompt") or ""
        resp = ex.get("response") or ex.get("solution") or ex.get("completion") or ""
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": instr},
             {"role": "assistant", "content": resp}],
            tokenize=False,
        )
        return {"text": text}

    ds = ds.map(to_messages, remove_columns=ds.column_names)

    def tokenize(ex):
        # chat template already adds special tokens -> don't add them again
        return tokenizer(ex["text"], truncation=True, max_length=args.seq_len,
                         add_special_tokens=False)

    return ds.map(tokenize, remove_columns=ds.column_names)


def main():
    args = parse_args()

    print(f"Loading {args.model_id} (device_map={args.device_map}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, dtype="auto", device_map=args.device_map, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    ds = build_calibration(tokenizer, args)

    # Fusion-compatible ignore list.
    ignore = ["lm_head", "re:.*mlp.gate$"]          # router gate stays bf16 (like QuantTrio)
    if not args.quantize_qkv:
        ignore += [                                  # keep QKV bf16 for the V1 fusion kernel
            "re:.*self_attn.q_proj$",
            "re:.*self_attn.k_proj$",
            "re:.*self_attn.v_proj$",
        ]
    print("Ignoring (kept in bf16):", ignore)

    recipe = [
        AWQModifier(),
        # W4A16_ASYM == 4-bit, group_size 128, asymmetric (zero-point) — matches
        # QuantTrio's {bits:4, group_size:128, zero_point:true}.
        QuantizationModifier(targets="Linear", scheme="W4A16_ASYM", ignore=ignore),
    ]

    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=args.seq_len,
        num_calibration_samples=args.calib_samples,
    )

    print(f"Saving compressed model to {args.output_dir} ...")
    model.save_pretrained(args.output_dir, save_compressed=True)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
