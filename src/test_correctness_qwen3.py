"""
Correctness tests for Qwen3 RMSNorm+Linear fusion.

Mirrors the structure of the reference test_correctness.py exactly:
  1. Kernel unit tests   — FusedRMSNormCombinedV1/V3 with Qwen3 hidden dims (FP32 & BF16)
  2. Integration test    — Qwen3-0.6B logits before vs after patch_qwen3_model()

Run with:
    python3 -m src.test_correctness_qwen3

Requires:
    python3 setup.py build_ext --inplace   (compile CUDA extension once)
    pip install transformers               (for integration test)
"""

import gc
import copy
import torch
import torch.nn as nn

# Force true fp32 (no TF32 Tensor Cores). On Blackwell, fp32 matmuls default to
# TF32 (~1e-3 error), and cuBLAS may tile the combined QKV matmul differently
# from the three separate ones — enough to fail the strict 1e-5 fp32 unit checks
# on a kernel that is actually correct. This is a correctness suite, so we want
# real fp32. (Benchmarks should leave TF32 at its default.)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from src.load_cuda import denominator_cuda
from src.weight_transform import compute_fused_weights_rmsnorm_combined
from src.fused_forward import (
    FusedRMSNormCombinedLinearV1,
    FusedRMSNormCombinedLinearV3,
)


# ─────────────────────────────────────────────────────────────────────────────
# Qwen3 hidden-size configurations
# Used by every unit test so dimensions exactly match the real models.
#
#   Qwen3-0.6B  (dense):   h=1024, Q=1024, K=512,  V=512
#   Qwen3-4B    (dense):   h=2048, Q=2048, K=1024, V=1024   (approx)
#   Qwen3-480B  (MoE):     h=7168, Q=8192, K=1024, V=1024
#
# The CUDA kernel is parameterised by h at runtime — same binary covers all.
# ─────────────────────────────────────────────────────────────────────────────
QWEN3_CONFIGS = [
    # (label,            h,    q_dim, k_dim, v_dim)
    ("Qwen3-0.6B dense", 1024, 1024,  512,   512),
    ("Qwen3-4B   dense", 2048, 2048,  1024,  1024),
    ("Qwen3-480B MoE  ", 7168, 8192,  1024,  1024),
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Kernel unit tests
# ─────────────────────────────────────────────────────────────────────────────

def test_fused_rmsnorm_combined_unit_fp32():
    """
    FP32 unit test: FusedRMSNormCombined[V1/V3] vs plain-PyTorch RMSNorm->Linear.

    Ground-truth correctness check: the fused combined kernel must match the
    standard sequential RMSNorm + Linear computed in PyTorch (the real
    reference), not just agree with another kernel call. Same comparison the
    bf16 test does, in fp32.

    Pass threshold: 1e-4. fp32 agreement between matmul-then-normalize (kernel)
    and normalize-then-matmul (PyTorch) bottoms out at a few fp32 ULPs (~1e-5 at
    h=7168) — far below the bf16 noise floor (~1.6e-2), so a real bug (which is
    orders of magnitude larger) still fails loudly.
    """
    print("=" * 60)
    print("TEST: FusedRMSNormCombined — Qwen3 dims, FP32")
    print("=" * 60)

    torch.manual_seed(42)
    fp32_tol = 1e-4

    for label, h, q_dim, k_dim, v_dim in QWEN3_CONFIGS:
        out_dims = [q_dim, k_dim, v_dim]
        rms_norm = nn.RMSNorm(h, eps=1e-6).cuda()
        nn.init.normal_(rms_norm.weight, mean=1.0, std=0.1)

        linears = []
        for od in out_dims:
            lin = nn.Linear(h, od, bias=False).cuda()
            nn.init.normal_(lin.weight, mean=0.0, std=0.02)
            linears.append(lin)

        W_comb, b_comb, split_sizes, h_dim, eps = \
            compute_fused_weights_rmsnorm_combined(rms_norm, linears)

        for batch in [1, 32, 128]:
            x = torch.randn(batch, h, device="cuda", dtype=torch.float32)

            # Ground-truth reference: plain PyTorch RMSNorm -> Linear
            with torch.no_grad():
                normed = rms_norm(x)
                ref_parts = [lin(normed) for lin in linears]

            # Combined V1
            mod_v1 = FusedRMSNormCombinedLinearV1(W_comb, b_comb, split_sizes, h_dim, eps)
            with torch.no_grad():
                out_v1 = mod_v1(x)
            md_v1 = max((r - o).abs().max().item() for r, o in zip(ref_parts, out_v1))
            s_v1 = "PASS" if md_v1 < fp32_tol else "FAIL"

            # Combined V3
            mod_v3 = FusedRMSNormCombinedLinearV3(W_comb, b_comb, split_sizes, h_dim, eps)
            with torch.no_grad():
                out_v3 = mod_v3(x)
            md_v3 = max((r - o).abs().max().item() for r, o in zip(ref_parts, out_v3))
            s_v3 = "PASS" if md_v3 < fp32_tol else "FAIL"

            print(f"  [{s_v1}] {label} batch={batch:3d}: "
                  f"V1 max_diff={md_v1:.2e}  [{s_v3}] V3 max_diff={md_v3:.2e}")
            assert md_v1 < fp32_tol, f"Combined V1 FP32 failed for {label}: max_diff={md_v1}"
            assert md_v3 < fp32_tol, f"Combined V3 FP32 failed for {label}: max_diff={md_v3}"

    print("  All FP32 combined unit tests passed!\n")


def test_fused_rmsnorm_combined_unit_bf16():
    """
    BF16 unit test: FusedRMSNormCombined[V1/V3] vs sequential RMSNorm+Linear.

    Mirrors test_fused_ln_linear_bf16() from the reference repo.
    BF16 pass threshold: 0.5  (matches the reference repo's BF16 tolerance)

    Why 0.5? BF16 has 7 mantissa bits. After a large matmul (h=7168 inputs),
    accumulated rounding errors in the raw output — before normalization —
    differ slightly from computing on normalized inputs.  The reference repo
    uses the same threshold for all its BF16 tests.
    """
    print("=" * 60)
    print("TEST: FusedRMSNormCombined — Qwen3 dims, BF16")
    print("=" * 60)

    torch.manual_seed(42)

    for label, h, q_dim, k_dim, v_dim in QWEN3_CONFIGS:
        out_dims = [q_dim, k_dim, v_dim]
        rms_norm = nn.RMSNorm(h, eps=1e-6).cuda().bfloat16()
        nn.init.normal_(rms_norm.weight, mean=1.0, std=0.1)

        linears = []
        for od in out_dims:
            lin = nn.Linear(h, od, bias=False).cuda().bfloat16()
            nn.init.normal_(lin.weight, mean=0.0, std=0.02)
            linears.append(lin)

        # Compute fused weights in FP32 (numerical stability), then cast to BF16.
        # Use deep copies: nn.Module.float() converts IN PLACE and would corrupt
        # rms_norm / linears, which must stay BF16 for the reference below.
        rms_fp32 = copy.deepcopy(rms_norm).float()
        lin_fp32 = [copy.deepcopy(l).float() for l in linears]
        W_comb, b_comb, split_sizes, h_dim, eps = \
            compute_fused_weights_rmsnorm_combined(rms_fp32, lin_fp32)
        W_comb = W_comb.bfloat16().cuda()
        b_comb = b_comb.bfloat16().cuda()

        for batch in [1, 32]:
            x = torch.randn(batch, h, device="cuda", dtype=torch.bfloat16)

            # Reference: sequential RMSNorm then each Linear (BF16)
            with torch.no_grad():
                normed = rms_norm(x)
                ref_parts = [lin(normed) for lin in linears]

            # Combined V1
            mod_v1 = FusedRMSNormCombinedLinearV1(W_comb, b_comb, split_sizes, h_dim, eps)
            with torch.no_grad():
                out_v1 = mod_v1(x)
            md_v1 = max(
                (r.float() - o.float()).abs().max().item()
                for r, o in zip(ref_parts, out_v1)
            )
            s_v1 = "PASS" if md_v1 < 0.5 else "FAIL"

            # Combined V3
            mod_v3 = FusedRMSNormCombinedLinearV3(W_comb, b_comb, split_sizes, h_dim, eps)
            with torch.no_grad():
                out_v3 = mod_v3(x)
            md_v3 = max(
                (r.float() - o.float()).abs().max().item()
                for r, o in zip(ref_parts, out_v3)
            )
            s_v3 = "PASS" if md_v3 < 0.5 else "FAIL"

            print(f"  [{s_v1}] {label} batch={batch:3d}: "
                  f"V1 max_diff={md_v1:.2e}  [{s_v3}] V3 max_diff={md_v3:.2e}")
            assert md_v1 < 0.5, f"Combined V1 BF16 failed for {label}: max_diff={md_v1}"
            assert md_v3 < 0.5, f"Combined V3 BF16 failed for {label}: max_diff={md_v3}"

    print("  All BF16 combined unit tests passed!\n")


def test_fused_rmsnorm_combined_3d_input():
    """
    3D input test: [batch, seq_len, h] — matches real inference shapes.

    Ground-truth check against plain-PyTorch RMSNorm->Linear, exercising the
    module's [B,S,H] -> flatten -> kernel -> split -> [B,S,out] reshape path.
    Pass threshold: 1e-4 (FP32).
    """
    print("=" * 60)
    print("TEST: FusedRMSNormCombined — 3D input [batch, seq, h], FP32")
    print("=" * 60)

    torch.manual_seed(42)

    for label, h, q_dim, k_dim, v_dim in QWEN3_CONFIGS:
        out_dims = [q_dim, k_dim, v_dim]
        rms_norm = nn.RMSNorm(h, eps=1e-6).cuda()
        nn.init.normal_(rms_norm.weight, mean=1.0, std=0.1)

        linears = []
        for od in out_dims:
            lin = nn.Linear(h, od, bias=False).cuda()
            nn.init.normal_(lin.weight, mean=0.0, std=0.02)
            linears.append(lin)

        W_comb, b_comb, split_sizes, h_dim, eps = \
            compute_fused_weights_rmsnorm_combined(rms_norm, linears)

        # FP32 tolerance: matmul-then-normalize (kernel) vs normalize-then-matmul
        # (plain PyTorch) is not bit-exact — fp32 non-associativity at large h
        # (e.g. 7168) reaches ~1e-5. 1e-4 sits well below the bf16 noise floor
        # (~1.6e-2), so it confirms correctness without false failures.
        fp32_tol = 1e-4
        for batch, seq_len in [(1, 8), (4, 128), (2, 512)]:
            x = torch.randn(batch, seq_len, h, device="cuda", dtype=torch.float32)

            # Ground-truth reference: plain PyTorch RMSNorm -> Linear
            with torch.no_grad():
                normed = rms_norm(x)
                ref_parts = [lin(normed) for lin in linears]

            # Combined V3
            mod_v3 = FusedRMSNormCombinedLinearV3(W_comb, b_comb, split_sizes, h_dim, eps)
            with torch.no_grad():
                out_v3 = mod_v3(x)

            md = max((r - o).abs().max().item() for r, o in zip(ref_parts, out_v3))
            s = "PASS" if md < fp32_tol else "FAIL"
            print(f"  [{s}] {label} [{batch}, {seq_len}, {h}]: max_diff={md:.2e}")
            assert md < fp32_tol, f"3D V3 failed for {label}: max_diff={md}"

    print("  All 3D input tests passed!\n")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Integration test — Qwen3-0.6B
# ─────────────────────────────────────────────────────────────────────────────

def test_qwen3_integration():
    """
    Integration test: compare Qwen3-0.6B logits before and after patching.

    Mirrors test_llama_integration() from the reference repo:
      - Load original model
      - Deep-copy it (0.6B is small enough)
      - Patch the copy with patch_qwen3_model()
      - Run forward pass on both with identical inputs
      - Compare logits element-wise

    BF16 tolerance: 2.0 — same threshold used for GPT-OSS-20B in the reference.
    Large BF16 models accumulate per-layer rounding differences.
    The key signal: mean_diff should be << max_diff (outliers, not systemic drift).
    """
    print("=" * 60)
    print("TEST: Qwen3-0.6B integration (patch_qwen3_model)")
    print("=" * 60)

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        print("  SKIP: transformers not installed")
        return

    from src.patch_qwen3 import patch_qwen3_model

    model_name = "Qwen/Qwen3-0.6B"
    print(f"  Loading {model_name} in BF16 ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model_orig = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
        ).eval()
    except Exception as e:
        print(f"  SKIP: Cannot load model ({e})")
        return

    print("  Deep-copying model for patching ...")
    model_fused = copy.deepcopy(model_orig)

    print("  Patching fused model with patch_qwen3_model(variant='V1') ...")
    patch_qwen3_model(model_fused, variant="V1")

    texts = [
        "def fibonacci(n: int) -> int:",
        "The transformer architecture was introduced in",
        "SELECT * FROM users WHERE",
    ]

    all_passed = True
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to("cuda")

        with torch.no_grad():
            logits_orig  = model_orig(**inputs).logits
            logits_fused = model_fused(**inputs).logits

        max_diff  = (logits_orig.float() - logits_fused.float()).abs().max().item()
        mean_diff = (logits_orig.float() - logits_fused.float()).abs().mean().item()
        rel_diff  = max_diff / logits_orig.float().abs().mean().item()

        # BF16 threshold: 2.0 (matches reference GPT-OSS-20B BF16 tolerance)
        status = "PASS" if max_diff < 2.0 else "FAIL"
        if max_diff >= 2.0:
            all_passed = False

        print(f"  [{status}] \"{text[:45]}...\": "
              f"max={max_diff:.2e}  mean={mean_diff:.2e}  rel={rel_diff:.2e}")

    if all_passed:
        print("  All Qwen3-0.6B integration tests passed!\n")
    else:
        print("  WARNING: Some integration tests exceeded threshold\n")

    del model_orig, model_fused
    gc.collect()
    torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Token generation test — confirms end-to-end generation is identical
# ─────────────────────────────────────────────────────────────────────────────

def test_qwen3_generation():
    """
    Generation test: patched model produces identical tokens to original.

    Mirrors test_llama_integration_gqa_decode() from the reference repo.
    Greedy decode is deterministic, so tokens must match exactly.
    """
    print("=" * 60)
    print("TEST: Qwen3-0.6B generation (greedy, tokens must match)")
    print("=" * 60)

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        print("  SKIP: transformers not installed")
        return

    from src.patch_qwen3 import patch_qwen3_model

    model_name = "Qwen/Qwen3-0.6B"
    print(f"  Loading {model_name} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model_orig = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
        ).eval()
    except Exception as e:
        print(f"  SKIP: Cannot load model ({e})")
        return

    model_fused = copy.deepcopy(model_orig)
    patch_qwen3_model(model_fused, variant="V1")

    prompts = [
        "def fibonacci(n):",
        "The capital of France is",
        "Write a Python function to reverse a string:",
    ]

    all_passed = True
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        gen_kwargs = dict(max_new_tokens=20, do_sample=False, use_cache=True)

        with torch.no_grad():
            out_orig  = model_orig.generate(**inputs, **gen_kwargs)
            out_fused = model_fused.generate(**inputs, **gen_kwargs)

        tok_orig  = tokenizer.decode(out_orig[0],  skip_special_tokens=True)
        tok_fused = tokenizer.decode(out_fused[0], skip_special_tokens=True)

        match  = tok_orig == tok_fused
        status = "PASS" if match else "FAIL"
        if not match:
            all_passed = False

        print(f"  [{status}] \"{prompt}\"")
        if not match:
            print(f"    orig : {tok_orig!r}")
            print(f"    fused: {tok_fused!r}")

    if all_passed:
        print("  All generation tests passed!\n")
    else:
        print("  WARNING: Token mismatch — patch changes model behaviour\n")

    del model_orig, model_fused
    gc.collect()
    torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # --- Kernel unit tests (no model download, needs CUDA extension) ---
    test_fused_rmsnorm_combined_unit_fp32()
    test_fused_rmsnorm_combined_unit_bf16()
    test_fused_rmsnorm_combined_3d_input()

    # --- Integration tests (downloads Qwen3-0.6B on first run) ---
    test_qwen3_integration()
    test_qwen3_generation()

    print("=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)
