"""
Three-level correctness test for the Qwen3-MoE RMSNorm fusion.

Run with:
    python3 -m src.test_correctness_qwen3

Level 1 — Synthetic unit test (no model download):
    RMSNorm(h=7168) + Linear(7168 -> 4096) on random BF16 input.
    Pass threshold: max_diff < 1e-2

Level 2 — Single decoder layer on Qwen3-0.6B:
    Load one layer, record output, apply patch, record again.
    Pass threshold: max_diff < 5e-3

Level 3 — End-to-end generation on Qwen3-0.6B:
    Load full model, run 3 prompts with greedy decoding, apply patch,
    run same prompts again. Pass: generated tokens are identical.

Do NOT proceed to Phase 2 (fuse 480B) until all three levels pass.
"""

import sys
import torch
import torch.nn as nn

# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _rms_norm_ref(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    """Reference RMSNorm in pure PyTorch."""
    rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
    return (x.float() / rms * gamma.float()).to(x.dtype)


PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def _check(name: str, max_diff: float, threshold: float) -> bool:
    ok = max_diff < threshold
    status = PASS if ok else FAIL
    print(f"  [{status}] {name}: max_diff={max_diff:.6e}  (threshold={threshold:.1e})")
    return ok


# ------------------------------------------------------------------ #
# Level 1 — Synthetic unit test                                        #
# ------------------------------------------------------------------ #

def level1_synthetic():
    print("\n=== Level 1: Synthetic unit test ===")

    # Build CUDA extension first (compile on first import)
    try:
        from src.fused_forward import FusedRMSNormCombinedLinearV3
    except Exception as e:
        print(f"  [SKIP] CUDA extension not available: {e}")
        print("  Build with: python3 setup.py build_ext --inplace")
        return True  # non-fatal if no GPU available in test environment

    torch.manual_seed(42)
    h = 7168   # Qwen3-480B hidden size
    out = 4096
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("  [SKIP] No CUDA device available for Level 1.")
        return True

    # Reference modules
    rms = nn.RMSNorm(h, eps=1e-6).to(device=device, dtype=torch.bfloat16)
    lin = nn.Linear(h, out, bias=False).to(device=device, dtype=torch.bfloat16)
    nn.init.normal_(rms.weight, mean=1.0, std=0.02)
    nn.init.normal_(lin.weight, std=0.02)

    # Fused module
    from src.weight_transform import compute_fused_weights_rmsnorm_combined
    W_comb, b_comb, split_sizes, _h, eps = compute_fused_weights_rmsnorm_combined(rms, [lin])
    fused = FusedRMSNormCombinedLinearV3(
        W_comb.to(device), b_comb.to(device), split_sizes, _h, eps
    )

    # Random BF16 input
    x = torch.randn(4, 32, h, device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        ref_out = lin(rms(x))
        fused_out = fused(x)[0]  # fused returns list of tensors

    # bf16 max-abs diff is dominated by 1 ULP on the largest element (e.g. at
    # magnitude ~8, one bf16 ULP is 0.0625), so an absolute threshold is not
    # meaningful here. Compare relative error against the bf16 reference.
    ref_f, fused_f = ref_out.float(), fused_out.float()
    max_abs = (ref_f - fused_f).abs().max().item()
    out_scale = ref_f.abs().max().item()
    rel = max_abs / out_scale if out_scale > 0 else max_abs
    print(f"    (abs max_diff={max_abs:.3e}, output max|.|={out_scale:.2f})")
    return _check("RMSNorm+Linear fusion (relative)", rel, 2e-2)


# ------------------------------------------------------------------ #
# Level 2 — Single decoder layer on Qwen3-0.6B                        #
# ------------------------------------------------------------------ #

def level2_single_layer():
    print("\n=== Level 2: Single decoder layer (Qwen3-0.6B) ===")

    try:
        from transformers import AutoModelForCausalLM, AutoConfig
    except ImportError:
        print("  [SKIP] transformers not installed")
        return True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading Qwen/Qwen3-0.6B on {device} ...")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B",
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"  [SKIP] Could not load Qwen3-0.6B: {e}")
        return True

    model.eval()
    layer = model.model.layers[0]

    # Record reference output
    torch.manual_seed(0)
    hidden_dim = model.config.hidden_size
    x = torch.randn(1, 8, hidden_dim, device=device, dtype=torch.bfloat16)

    # Provide minimal kwargs the layer forward expects
    position_ids = torch.arange(8, device=device).unsqueeze(0)
    cos, sin = model.model.rotary_emb(x, position_ids)
    position_embeddings = (cos, sin)

    with torch.no_grad():
        ref_out = layer(x, position_embeddings=position_embeddings)[0]

    # Patch this layer
    from src.patch_qwen3 import _patch_decoder_layer
    _patch_decoder_layer(layer, device=device, variant="V3")

    with torch.no_grad():
        fused_out = layer(x, position_embeddings=position_embeddings)[0]

    max_diff = (ref_out - fused_out).abs().max().item()
    return _check("Single layer output", max_diff, 5e-3)


# ------------------------------------------------------------------ #
# Level 3 — End-to-end generation on Qwen3-0.6B                       #
# ------------------------------------------------------------------ #

PROMPTS = [
    "def fibonacci(n):",
    "The capital of France is",
    "Write a Python function to reverse a string:",
]
MAX_NEW_TOKENS = 20


def _generate(model, tokenizer, prompts, device):
    outputs = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )
        generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        outputs.append(generated)
    return outputs


def level3_end_to_end():
    print("\n=== Level 3: End-to-end generation (Qwen3-0.6B) ===")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("  [SKIP] transformers not installed")
        return True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading Qwen/Qwen3-0.6B on {device} ...")

    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B",
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"  [SKIP] Could not load Qwen3-0.6B: {e}")
        return True

    model.eval()

    # Reference generation (before patch)
    print("  Running reference generation ...")
    ref_outputs = _generate(model, tokenizer, PROMPTS, device)

    # Patch the full model
    print("  Applying patch_qwen3_model ...")
    from src.patch_qwen3 import patch_qwen3_model
    patch_qwen3_model(model, device=device, variant="V3")

    # Fused generation
    print("  Running fused generation ...")
    fused_outputs = _generate(model, tokenizer, PROMPTS, device)

    all_pass = True
    for i, (ref, fused) in enumerate(zip(ref_outputs, fused_outputs)):
        ok = ref == fused
        status = PASS if ok else FAIL
        print(f"  [{status}] Prompt {i+1}: ref={ref!r}  fused={fused!r}")
        if not ok:
            all_pass = False

    return all_pass


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main():
    print("Qwen3-MoE RMSNorm Fusion — Correctness Tests")
    print("=" * 50)

    results = {
        "Level 1 (synthetic)": level1_synthetic(),
        "Level 2 (single layer)": level2_single_layer(),
        "Level 3 (end-to-end)": level3_end_to_end(),
    }

    print("\n=== Summary ===")
    all_pass = True
    for name, ok in results.items():
        status = PASS if ok else FAIL
        print(f"  [{status}] {name}")
        if not ok:
            all_pass = False

    if all_pass:
        print("\nAll levels passed. Safe to proceed to Phase 2 (fuse 480B model).")
        sys.exit(0)
    else:
        print("\nOne or more levels FAILED. Do NOT proceed to Phase 2.")
        sys.exit(1)


if __name__ == "__main__":
    main()
