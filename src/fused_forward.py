"""
Fused LayerNorm+Linear forward pass using concurrent CUDA streams.

The matmul (x @ W_new.T) runs on the default stream while the denominator
kernel (v(x) = ||x - mean(x)||_2) runs on a separate stream concurrently.

The exact LayerNorm standard deviation is: std = sqrt(v^2/h + eps)
where v = ||x - mean(x)||_2, h = hidden_dim, eps = LN epsilon.
"""

import os
import torch
import torch.nn.functional as F
from src.load_cuda import denominator_cuda

# Optional NVTX annotations for Nsight profiling
_USE_NVTX = os.environ.get("FUSED_LN_NVTX", "0") == "1"


def _nvtx_range(name):
    """Context manager for NVTX range annotation (no-op if disabled)."""
    if _USE_NVTX:
        return torch.cuda.nvtx.range(name)
    return _NullContext()


class _NullContext:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


class FusedLNLinear(torch.nn.Module):
    """
    Replaces a sequential LayerNorm -> Linear with a single fused operation.

    Forward:
        raw = x @ W_new.T           (default stream: matmul)
        v   = denominator(x)        (separate stream: lightweight reduction)
        std = sqrt(v^2/h + eps)     (exact LN std)
        out = raw / std[..., None] + b_new
    """

    def __init__(self, W_new: torch.Tensor, b_new: torch.Tensor,
                 denom_stream: torch.cuda.Stream, h: int, eps: float):
        super().__init__()
        self.register_buffer("W_new", W_new)
        self.register_buffer("b_new", b_new)
        self.denom_stream = denom_stream
        self.h = h
        self.eps = eps
        # Pre-allocate events to avoid per-call overhead
        self._input_ready = torch.cuda.Event(enable_timing=False)
        self._denom_done = torch.cuda.Event(enable_timing=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with _nvtx_range("FusedLNLinear_V0"):
            orig_shape = x.shape
            x_2d = x.reshape(-1, x.size(-1))

            default_stream = torch.cuda.current_stream()

            # Signal that input is ready for the denominator stream
            self._input_ready.record(default_stream)
            self.denom_stream.wait_event(self._input_ready)

            # Launch denominator on separate stream
            with torch.cuda.stream(self.denom_stream):
                v = denominator_cuda.compute_denominator(x_2d)

            # Matmul on default stream (concurrent with denominator)
            raw_output = F.linear(x_2d, self.W_new)

            # Wait for denominator
            self._denom_done.record(self.denom_stream)
            default_stream.wait_event(self._denom_done)

            # Exact LayerNorm std
            std = torch.sqrt(v * v / self.h + self.eps)
            output = raw_output / std.unsqueeze(-1) + self.b_new

            out_shape = orig_shape[:-1] + (output.size(-1),)
            return output.reshape(out_shape)


class FusedLNLinearV1(torch.nn.Module):
    """
    V1: Fused LN+Linear with fused denominator+normalize kernel (no streams).

    Forward:
        raw = F.linear(x, W_new)                    (cuBLAS matmul)
        denominator_normalize(x, raw, b_new, h, eps) (single kernel: denom + normalize in-place)
    """

    def __init__(self, W_new: torch.Tensor, b_new: torch.Tensor, h: int, eps: float):
        super().__init__()
        self.register_buffer("W_new", W_new)
        self.register_buffer("b_new", b_new)
        self.h = h
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with _nvtx_range("FusedLNLinear_V1"):
            orig_shape = x.shape
            x_2d = x.reshape(-1, x.size(-1))

            raw_output = F.linear(x_2d, self.W_new)
            denominator_cuda.denominator_normalize(x_2d, raw_output, self.b_new, self.h, self.eps)

            out_shape = orig_shape[:-1] + (raw_output.size(-1),)
            return raw_output.reshape(out_shape)


class FusedLNLinearV3(torch.nn.Module):
    """
    V3: Fused LN+Linear with Welford + fused normalize + 512 threads (no streams).

    Forward:
        raw = F.linear(x, W_new)
        denominator_normalize_welford_512(x, raw, b_new, h, eps)
    """

    def __init__(self, W_new: torch.Tensor, b_new: torch.Tensor, h: int, eps: float):
        super().__init__()
        self.register_buffer("W_new", W_new)
        self.register_buffer("b_new", b_new)
        self.h = h
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with _nvtx_range("FusedLNLinear_V3"):
            orig_shape = x.shape
            x_2d = x.reshape(-1, x.size(-1))

            raw_output = F.linear(x_2d, self.W_new)
            denominator_cuda.denominator_normalize_welford_512(x_2d, raw_output, self.b_new, self.h, self.eps)

            out_shape = orig_shape[:-1] + (raw_output.size(-1),)
            return raw_output.reshape(out_shape)


class FusedRMSNormLinearV1(torch.nn.Module):
    """
    V1: Fused RMSNorm+Linear with fused rms+normalize kernel (no streams).

    RMSNorm: rms(x) = sqrt(mean(x^2) + eps), then normalize by 1/rms.
    Weight gamma is absorbed into W_new during weight precomputation.
    """

    def __init__(self, W_new: torch.Tensor, b_new: torch.Tensor, h: int, eps: float):
        super().__init__()
        self.register_buffer("W_new", W_new)
        self.register_buffer("b_new", b_new)
        self.h = h
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with _nvtx_range("FusedRMSNormLinear_V1"):
            orig_shape = x.shape
            x_2d = x.reshape(-1, x.size(-1))

            raw_output = F.linear(x_2d, self.W_new)
            denominator_cuda.rmsnorm_normalize(x_2d, raw_output, self.b_new, self.h, self.eps)

            out_shape = orig_shape[:-1] + (raw_output.size(-1),)
            return raw_output.reshape(out_shape)


class FusedRMSNormLinearV3(torch.nn.Module):
    """
    V3: Fused RMSNorm+Linear with fused rms+normalize + 512 threads (no streams).
    """

    def __init__(self, W_new: torch.Tensor, b_new: torch.Tensor, h: int, eps: float):
        super().__init__()
        self.register_buffer("W_new", W_new)
        self.register_buffer("b_new", b_new)
        self.h = h
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with _nvtx_range("FusedRMSNormLinear_V3"):
            orig_shape = x.shape
            x_2d = x.reshape(-1, x.size(-1))

            raw_output = F.linear(x_2d, self.W_new)
            denominator_cuda.rmsnorm_normalize_512(x_2d, raw_output, self.b_new, self.h, self.eps)

            out_shape = orig_shape[:-1] + (raw_output.size(-1),)
            return raw_output.reshape(out_shape)


class FusedRMSNormCombinedLinearV1(torch.nn.Module):
    """
    V1: Fused RMSNorm + combined Linear projections (256 threads).

    Replaces multiple RMSNorm+Linear pairs that share the same norm (e.g. Q/K/V
    or gate/up) with a single matmul + single normalize kernel call.
    Returns a list of tensors split along the output dimension.
    """

    def __init__(self, W_combined: torch.Tensor, b_combined: torch.Tensor,
                 split_sizes: list[int], h: int, eps: float):
        super().__init__()
        self.register_buffer("W_combined", W_combined)
        self.register_buffer("b_combined", b_combined)
        self.split_sizes = split_sizes
        self.h = h
        self.eps = eps

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        with _nvtx_range("FusedRMSNormCombinedLinear_V1"):
            orig_shape = x.shape
            x_2d = x.reshape(-1, x.size(-1))

            raw_output = F.linear(x_2d, self.W_combined)
            denominator_cuda.rmsnorm_normalize(x_2d, raw_output, self.b_combined, self.h, self.eps)

            # Split along output dim and make contiguous (split on dim=-1 of
            # row-major 2D produces non-contiguous views)
            parts = torch.split(raw_output, self.split_sizes, dim=-1)
            batch_shape = orig_shape[:-1]
            return [p.contiguous().reshape(*batch_shape, p.size(-1)) for p in parts]


class FusedRMSNormCombinedLinearV3(torch.nn.Module):
    """
    V3: Fused RMSNorm + combined Linear projections (512 threads).
    """

    def __init__(self, W_combined: torch.Tensor, b_combined: torch.Tensor,
                 split_sizes: list[int], h: int, eps: float):
        super().__init__()
        self.register_buffer("W_combined", W_combined)
        self.register_buffer("b_combined", b_combined)
        self.split_sizes = split_sizes
        self.h = h
        self.eps = eps

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        with _nvtx_range("FusedRMSNormCombinedLinear_V3"):
            orig_shape = x.shape
            x_2d = x.reshape(-1, x.size(-1))

            raw_output = F.linear(x_2d, self.W_combined)
            denominator_cuda.rmsnorm_normalize_512(x_2d, raw_output, self.b_combined, self.h, self.eps)

            parts = torch.split(raw_output, self.split_sizes, dim=-1)
            batch_shape = orig_shape[:-1]
            return [p.contiguous().reshape(*batch_shape, p.size(-1)) for p in parts]


class FusedRMSNormSwiGLUV1(torch.nn.Module):
    """
    V1: Fused RMSNorm + combined gate+up matmul + SwiGLU (256 threads).

    Input: x [batch, h] -> matmul -> [batch, 2*intermediate]
    Then fused kernel: RMS norm on x, normalize gate/up, SiLU(gate)*up
    Output: [batch, intermediate]
    """

    def __init__(self, W_combined: torch.Tensor, b_combined: torch.Tensor,
                 intermediate: int, h: int, eps: float):
        super().__init__()
        self.register_buffer("W_combined", W_combined)
        self.register_buffer("b_combined", b_combined)
        self.intermediate = intermediate
        self.h = h
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with _nvtx_range("FusedRMSNormSwiGLU_V1"):
            orig_shape = x.shape
            x_2d = x.reshape(-1, x.size(-1))

            raw_output = F.linear(x_2d, self.W_combined)
            output = torch.empty(x_2d.size(0), self.intermediate,
                                 dtype=x_2d.dtype, device=x_2d.device)
            denominator_cuda.rmsnorm_swiglu(
                x_2d, raw_output, output, self.b_combined,
                self.h, self.intermediate, self.eps
            )
            return output.reshape(*orig_shape[:-1], self.intermediate)


class FusedRMSNormSwiGLUV3(torch.nn.Module):
    """
    V3: Fused RMSNorm + combined gate+up matmul + SwiGLU (512 threads).
    """

    def __init__(self, W_combined: torch.Tensor, b_combined: torch.Tensor,
                 intermediate: int, h: int, eps: float):
        super().__init__()
        self.register_buffer("W_combined", W_combined)
        self.register_buffer("b_combined", b_combined)
        self.intermediate = intermediate
        self.h = h
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with _nvtx_range("FusedRMSNormSwiGLU_V3"):
            orig_shape = x.shape
            x_2d = x.reshape(-1, x.size(-1))

            raw_output = F.linear(x_2d, self.W_combined)
            output = torch.empty(x_2d.size(0), self.intermediate,
                                 dtype=x_2d.dtype, device=x_2d.device)
            denominator_cuda.rmsnorm_swiglu_512(
                x_2d, raw_output, output, self.b_combined,
                self.h, self.intermediate, self.eps
            )
            return output.reshape(*orig_shape[:-1], self.intermediate)


def fused_rmsnorm_linear_forward_v1(
    x: torch.Tensor,
    W_new: torch.Tensor,
    b_new: torch.Tensor,
    h: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """V1: Functional fused RMSNorm+Linear (256 threads)."""
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.size(-1))

    raw_output = F.linear(x_2d, W_new)
    denominator_cuda.rmsnorm_normalize(x_2d, raw_output, b_new, h, eps)

    out_shape = orig_shape[:-1] + (raw_output.size(-1),)
    return raw_output.reshape(out_shape)


def fused_rmsnorm_linear_forward_v3(
    x: torch.Tensor,
    W_new: torch.Tensor,
    b_new: torch.Tensor,
    h: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """V3: Functional fused RMSNorm+Linear (512 threads)."""
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.size(-1))

    raw_output = F.linear(x_2d, W_new)
    denominator_cuda.rmsnorm_normalize_512(x_2d, raw_output, b_new, h, eps)

    out_shape = orig_shape[:-1] + (raw_output.size(-1),)
    return raw_output.reshape(out_shape)


def fused_ln_linear_forward_v1(
    x: torch.Tensor,
    W_new: torch.Tensor,
    b_new: torch.Tensor,
    h: int,
    eps: float = 1e-5,
) -> torch.Tensor:
    """V1: Functional fused LN+Linear with fused denominator+normalize (no streams)."""
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.size(-1))

    raw_output = F.linear(x_2d, W_new)
    denominator_cuda.denominator_normalize(x_2d, raw_output, b_new, h, eps)

    out_shape = orig_shape[:-1] + (raw_output.size(-1),)
    return raw_output.reshape(out_shape)


def fused_ln_linear_forward_v3(
    x: torch.Tensor,
    W_new: torch.Tensor,
    b_new: torch.Tensor,
    h: int,
    eps: float = 1e-5,
) -> torch.Tensor:
    """V3: Functional fused LN+Linear with Welford + normalize + 512 threads."""
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.size(-1))

    raw_output = F.linear(x_2d, W_new)
    denominator_cuda.denominator_normalize_welford_512(x_2d, raw_output, b_new, h, eps)

    out_shape = orig_shape[:-1] + (raw_output.size(-1),)
    return raw_output.reshape(out_shape)


def fused_ln_linear_forward(
    x: torch.Tensor,
    W_new: torch.Tensor,
    b_new: torch.Tensor,
    denom_stream: torch.cuda.Stream,
    h: int,
    eps: float = 1e-5,
    _input_ready: torch.cuda.Event = None,
    _denom_done: torch.cuda.Event = None,
) -> torch.Tensor:
    """Functional version of fused LayerNorm+Linear forward."""
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.size(-1))

    default_stream = torch.cuda.current_stream()

    if _input_ready is None:
        _input_ready = torch.cuda.Event(enable_timing=False)
    if _denom_done is None:
        _denom_done = torch.cuda.Event(enable_timing=False)

    _input_ready.record(default_stream)
    denom_stream.wait_event(_input_ready)

    with torch.cuda.stream(denom_stream):
        v = denominator_cuda.compute_denominator(x_2d)

    raw_output = F.linear(x_2d, W_new)

    _denom_done.record(denom_stream)
    default_stream.wait_event(_denom_done)

    std = torch.sqrt(v * v / h + eps)
    output = raw_output / std.unsqueeze(-1) + b_new

    out_shape = orig_shape[:-1] + (output.size(-1),)
    return output.reshape(out_shape)
