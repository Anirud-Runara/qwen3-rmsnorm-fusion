#!/usr/bin/env python3
"""
Probe: can we load a *modelopt* NVFP4 Qwen3-MoE checkpoint in HF transformers,
keep it 4-bit in memory, and reach the q/k/v_proj weights for fusion?

WHY THIS EXISTS
  The benchmark plan is: load an NVFP4 checkpoint in HF transformers, monkey-patch
  the RMSNorm+QKV fusion kernel onto it (which requires dequantizing the fused
  layers to BF16), and compare vs the unpatched NVFP4 baseline.

  But nvidia/Qwen3-Coder-480B-A35B-Instruct-NVFP4 is `quant_method: modelopt`
  (TensorRT-LLM oriented), NOT `compressed-tensors`. Three things must hold for
  the plan to work, and all three are unknowns until we try:

    1. LOADABLE  — does AutoModelForCausalLM.from_pretrained() actually load it
                   in eager HF transformers (maybe only with nvidia-modelopt
                   installed / its "unified HF checkpoint" restore path)?
    2. STILL 4-BIT — does it stay packed (real memory saving), or does it
                   dequantize to BF16 on load (→ 480B balloons to ~960GB → OOM)?
    3. DEQUANTABLE — can we get a dense [out, in] weight out of q_proj so the
                   fusion's compute_fused_weights can fold gamma into it?

  Run this on the SMALL sibling (nvidia/Qwen3-30B-A3B-NVFP4, same format+arch)
  on ONE GPU before committing to the 480B.

USAGE
    # plain transformers first:
    python3 quantization/probe_nvfp4_loading.py --model-id nvidia/Qwen3-30B-A3B-NVFP4

    # if that fails, install nvidia-modelopt's HF integration and retry:
    pip install "nvidia-modelopt[hf]"
    python3 quantization/probe_nvfp4_loading.py --model-id nvidia/Qwen3-30B-A3B-NVFP4 --use-modelopt

WHAT TO READ OFF THE OUTPUT
    - "LOAD: ok"                  → step 1 passes
    - q_proj type + buffers       → tells us the dequant API (compressed-tensors
                                    vs modelopt vs plain) and whether .weight is dense
    - "GPU mem ... GiB" vs a BF16 estimate → step 2 (stayed 4-bit if << BF16 size)
    - "DEQUANT: produced dense [out,in] bf16" → step 3 passes; copy that method
                                    into compute_fused_weights
"""

import argparse
import importlib
import torch


def parse_args():
    p = argparse.ArgumentParser(description="Probe modelopt NVFP4 loadability in HF transformers")
    p.add_argument("--model-id", default="nvidia/Qwen3-30B-A3B-NVFP4",
                   help="Small modelopt-NVFP4 Qwen3-MoE to validate before the 480B")
    p.add_argument("--use-modelopt", action="store_true",
                   help="Import modelopt.torch.quantization first (registers/restores modelopt quant)")
    p.add_argument("--layer-idx", type=int, default=0)
    return p.parse_args()


def _gpu_mem_gib() -> float:
    if not torch.cuda.is_available():
        return float("nan")
    return max(torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())) / 1024**3


def describe_module(mod) -> None:
    """Print the module class + its params/buffers so we can see the quant layout."""
    print(f"    class: {type(mod).__module__}.{type(mod).__name__}")
    for name, t in list(mod.named_parameters(recurse=False)) + list(mod.named_buffers(recurse=False)):
        print(f"    {name:<24} {tuple(t.shape)!s:<22} {t.dtype}")
    w = getattr(mod, "weight", None)
    if w is not None:
        print(f"    .weight present: shape={tuple(w.shape)} dtype={w.dtype} "
              f"(dense BF16/FP16 ⇒ already de-quantized in memory)")
    else:
        print("    .weight absent ⇒ packed/quantized layout (good for memory, needs dequant)")


def try_dequant(q_proj) -> None:
    """Attempt every dequant path we know; report which one yields a dense weight."""
    # Path A — compressed-tensors
    try:
        from compressed_tensors.linear.compressed_linear import CompressedLinear  # noqa
        if isinstance(q_proj, CompressedLinear):
            W = q_proj.compressor.decompress_module(q_proj)
            print(f"  DEQUANT: compressed-tensors → dense {tuple(W.shape)} {W.dtype}")
            return
    except Exception as e:
        print(f"  DEQUANT[compressed-tensors]: n/a ({type(e).__name__}: {e})")

    # Path B — modelopt (weights kept; dequantize via its quantizer state)
    try:
        import modelopt.torch.quantization as mtq  # noqa
        # modelopt typically exposes the real weight on .weight and a TensorQuantizer
        # (.weight_quantizer). Fake-quant means .weight is already dense.
        if hasattr(q_proj, "weight") and q_proj.weight is not None and q_proj.weight.dim() == 2:
            W = q_proj.weight.data.to(torch.bfloat16)
            print(f"  DEQUANT: modelopt .weight is dense → {tuple(W.shape)} {W.dtype} "
                  f"(fake-quant; NOTE: memory NOT saved if all layers like this)")
            return
        print("  DEQUANT[modelopt]: .weight not a dense 2-D tensor — needs manual e2m1 unpack")
    except Exception as e:
        print(f"  DEQUANT[modelopt]: n/a ({type(e).__name__}: {e})")

    # Path C — plain Linear
    if hasattr(q_proj, "weight") and q_proj.weight is not None and q_proj.weight.dim() == 2:
        print(f"  DEQUANT: plain dense .weight → {tuple(q_proj.weight.shape)} {q_proj.weight.dtype}")
        return

    print("  DEQUANT: NO known path produced a dense weight — inspect the buffers above "
          "(look for weight_packed / weight_scale / amax → manual NVFP4 unpack required)")


def main():
    args = parse_args()
    print(f"PyTorch {torch.__version__} | CUDA {torch.version.cuda} | "
          f"GPUs {torch.cuda.device_count() if torch.cuda.is_available() else 0}")

    if args.use_modelopt:
        try:
            import modelopt.torch.quantization  # noqa  (registers modelopt quant w/ transformers)
            print("modelopt.torch.quantization imported.")
        except Exception as e:
            print(f"WARNING: could not import modelopt ({e}); install 'nvidia-modelopt[hf]'")

    from transformers import AutoModelForCausalLM

    print(f"\nLoading {args.model_id} (device_map=auto, dtype=auto) ...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, dtype="auto", device_map="auto", trust_remote_code=True,
        )
        model.eval()
        print("LOAD: ok")
    except Exception as e:
        print(f"LOAD: FAILED ({type(e).__name__}: {e})")
        print("→ If this says 'Unknown quantization method modelopt', retry with "
              "--use-modelopt after `pip install nvidia-modelopt[hf]`.")
        print("→ If it still fails, the checkpoint is not HF-eager loadable; switch to a "
              "compressed-tensors NVFP4 checkpoint (llm-compressor) for the HF benchmark.")
        return

    print(f"\nGPU mem (max over GPUs): {_gpu_mem_gib():.1f} GiB  "
          f"(compare to a BF16 estimate: stayed 4-bit if much smaller)")

    layer = model.model.layers[args.layer_idx]
    print(f"\nq_proj of layer {args.layer_idx}:")
    describe_module(layer.self_attn.q_proj)
    print(f"\ninput_layernorm of layer {args.layer_idx}:")
    describe_module(layer.input_layernorm)

    print("\nTrying dequant paths on q_proj ...")
    try_dequant(layer.self_attn.q_proj)

    print("\nTiny forward (1×8 tokens) ...")
    try:
        ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], device=next(model.parameters()).device)
        with torch.no_grad():
            out = model(ids)
        print(f"FORWARD: ok — logits {tuple(out.logits.shape)} {out.logits.dtype}")
    except Exception as e:
        print(f"FORWARD: FAILED ({type(e).__name__}: {e})")

    print("\n=== VERDICT ===")
    print("If LOAD+FORWARD ok AND a DEQUANT path produced a dense bf16 weight AND mem stayed "
          "well below BF16 size → the HF monkey-patch plan is viable on this format; wire that "
          "dequant path into compute_fused_weights. Otherwise, fall back to compressed-tensors NVFP4.")


if __name__ == "__main__":
    main()
