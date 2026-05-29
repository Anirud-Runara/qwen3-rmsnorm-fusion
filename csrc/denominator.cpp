#include <torch/extension.h>

// V0: Original two-pass denominator
torch::Tensor compute_denominator_cuda(
    torch::Tensor x,
    c10::optional<int64_t> stream_ptr
);

// V1: Fused denominator + normalize (in-place)
void denominator_normalize_cuda(
    torch::Tensor x,
    torch::Tensor raw_output,
    torch::Tensor b_new,
    int h,
    float eps
);

// V2: Welford single-pass denominator
torch::Tensor compute_denominator_welford_cuda(
    torch::Tensor x,
    c10::optional<int64_t> stream_ptr
);

// V3: Welford + fused normalize + 512 threads
void denominator_normalize_welford_512_cuda(
    torch::Tensor x,
    torch::Tensor raw_output,
    torch::Tensor b_new,
    int h,
    float eps
);

// RMSNorm V1: fused normalize (256 threads)
void rmsnorm_normalize_cuda(
    torch::Tensor x,
    torch::Tensor raw_output,
    torch::Tensor b_new,
    int h,
    float eps
);

// RMSNorm V3: fused normalize (512 threads)
void rmsnorm_normalize_512_cuda(
    torch::Tensor x,
    torch::Tensor raw_output,
    torch::Tensor b_new,
    int h,
    float eps
);

// RMSNorm+SwiGLU V1: fused normalize + SiLU + multiply (256 threads)
void rmsnorm_swiglu_cuda(
    torch::Tensor x,
    torch::Tensor raw_output,
    torch::Tensor output,
    torch::Tensor b_new,
    int h,
    int intermediate,
    float eps
);

// RMSNorm+SwiGLU V3: fused normalize + SiLU + multiply (512 threads)
void rmsnorm_swiglu_512_cuda(
    torch::Tensor x,
    torch::Tensor raw_output,
    torch::Tensor output,
    torch::Tensor b_new,
    int h,
    int intermediate,
    float eps
);

// Python wrappers
torch::Tensor compute_denominator(
    torch::Tensor x,
    c10::optional<int64_t> stream_ptr = c10::nullopt
) {
    return compute_denominator_cuda(x, stream_ptr);
}

void denominator_normalize(
    torch::Tensor x,
    torch::Tensor raw_output,
    torch::Tensor b_new,
    int h,
    float eps
) {
    denominator_normalize_cuda(x, raw_output, b_new, h, eps);
}

torch::Tensor compute_denominator_welford(
    torch::Tensor x,
    c10::optional<int64_t> stream_ptr = c10::nullopt
) {
    return compute_denominator_welford_cuda(x, stream_ptr);
}

void denominator_normalize_welford_512(
    torch::Tensor x,
    torch::Tensor raw_output,
    torch::Tensor b_new,
    int h,
    float eps
) {
    denominator_normalize_welford_512_cuda(x, raw_output, b_new, h, eps);
}

void rmsnorm_normalize(
    torch::Tensor x,
    torch::Tensor raw_output,
    torch::Tensor b_new,
    int h,
    float eps
) {
    rmsnorm_normalize_cuda(x, raw_output, b_new, h, eps);
}

void rmsnorm_normalize_512(
    torch::Tensor x,
    torch::Tensor raw_output,
    torch::Tensor b_new,
    int h,
    float eps
) {
    rmsnorm_normalize_512_cuda(x, raw_output, b_new, h, eps);
}

void rmsnorm_swiglu(
    torch::Tensor x,
    torch::Tensor raw_output,
    torch::Tensor output,
    torch::Tensor b_new,
    int h,
    int intermediate,
    float eps
) {
    rmsnorm_swiglu_cuda(x, raw_output, output, b_new, h, intermediate, eps);
}

void rmsnorm_swiglu_512(
    torch::Tensor x,
    torch::Tensor raw_output,
    torch::Tensor output,
    torch::Tensor b_new,
    int h,
    int intermediate,
    float eps
) {
    rmsnorm_swiglu_512_cuda(x, raw_output, output, b_new, h, intermediate, eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_denominator", &compute_denominator,
          "V0: Compute denominator v(x) = ||x - mean(x)||_2 per row (two-pass)",
          py::arg("x"),
          py::arg("stream_ptr") = py::none());

    m.def("denominator_normalize", &denominator_normalize,
          "V1: Fused denominator + normalize raw_output in-place (two-pass, no streams)",
          py::arg("x"),
          py::arg("raw_output"),
          py::arg("b_new"),
          py::arg("h"),
          py::arg("eps"));

    m.def("compute_denominator_welford", &compute_denominator_welford,
          "V2: Welford single-pass denominator v(x) = ||x - mean(x)||_2 per row",
          py::arg("x"),
          py::arg("stream_ptr") = py::none());

    m.def("denominator_normalize_welford_512", &denominator_normalize_welford_512,
          "V3: Welford + fused normalize + 512 threads (best combined variant)",
          py::arg("x"),
          py::arg("raw_output"),
          py::arg("b_new"),
          py::arg("h"),
          py::arg("eps"));

    m.def("rmsnorm_normalize", &rmsnorm_normalize,
          "RMSNorm V1: Fused RMSNorm normalize in-place (256 threads)",
          py::arg("x"),
          py::arg("raw_output"),
          py::arg("b_new"),
          py::arg("h"),
          py::arg("eps"));

    m.def("rmsnorm_normalize_512", &rmsnorm_normalize_512,
          "RMSNorm V3: Fused RMSNorm normalize in-place (512 threads)",
          py::arg("x"),
          py::arg("raw_output"),
          py::arg("b_new"),
          py::arg("h"),
          py::arg("eps"));

    m.def("rmsnorm_swiglu", &rmsnorm_swiglu,
          "RMSNorm + SwiGLU V1: fused normalize + SiLU + multiply (256 threads)",
          py::arg("x"),
          py::arg("raw_output"),
          py::arg("output"),
          py::arg("b_new"),
          py::arg("h"),
          py::arg("intermediate"),
          py::arg("eps"));

    m.def("rmsnorm_swiglu_512", &rmsnorm_swiglu_512,
          "RMSNorm + SwiGLU V3: fused normalize + SiLU + multiply (512 threads)",
          py::arg("x"),
          py::arg("raw_output"),
          py::arg("output"),
          py::arg("b_new"),
          py::arg("h"),
          py::arg("intermediate"),
          py::arg("eps"));
}
