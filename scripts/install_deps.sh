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
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate
pip install huggingface_hub[cli] hf_transfer datasets

echo "=== Compiling CUDA denominator extension ==="
# Must be run on a machine with a CUDA-capable GPU and CUDA toolkit installed.
python3 setup.py build_ext --inplace

echo "=== All dependencies installed ==="
