#!/usr/bin/env bash
# Install all Python dependencies for the qwen3-rmsnorm-fusion project.
#
# Usage:
#   bash scripts/install_deps.sh
#
# Run this on any machine that will run correctness tests or the fusion script.
# GPU machines also need the CUDA extension compiled (see the final step below).
set -euo pipefail

echo "=== Installing core Python dependencies ==="
# PyTorch: Blackwell GPUs (sm_120, e.g. RTX Pro 6000) require CUDA 12.8+ wheels.
# Most GPU base images already ship a CUDA-capable PyTorch — reinstalling can
# downgrade it to a build without Blackwell kernels, so only install if missing.
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "PyTorch already present: $(python3 -c 'import torch; print(torch.__version__, torch.version.cuda)')"
else
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
fi
pip install transformers accelerate
pip install huggingface_hub[cli] hf_transfer datasets

echo "=== Compiling CUDA denominator extension ==="
# Must be run on a machine with a CUDA-capable GPU and CUDA toolkit installed.
# Non-fatal: if torch's CUDA version differs from the system toolkit (e.g. a
# cu128 torch wheel on a CUDA 13.0 box), this AOT build trips torch's version
# check. That's OK — the kernel is JIT-compiled on first import via
# src/load_cuda.py, which bypasses the check. The tests rely on the JIT path.
python3 setup.py build_ext --inplace || {
    echo "WARNING: AOT build_ext failed (likely a torch/toolkit CUDA version mismatch)."
    echo "This is non-fatal — the kernel will JIT-compile on first import."
}

echo "=== All dependencies installed ==="
