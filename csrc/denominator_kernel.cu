#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>

// 256 threads per block, one block per row
constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;

// Warp-level reduction using shuffle
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Block-level reduction via shared memory
__device__ float block_reduce_sum(float val, float* smem) {
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        smem[warp_id] = val;
    }
    __syncthreads();

    // First warp reduces across warps
    int num_warps = BLOCK_SIZE / WARP_SIZE;  // 8 warps
    if (warp_id == 0) {
        val = (lane < num_warps) ? smem[lane] : 0.0f;
        val = warp_reduce_sum(val);
    }
    __syncthreads();
    return val;  // Only valid in thread 0
}

// FP32 kernel: one block per row, vectorized float4 loads
__global__ void denominator_fp32_kernel(
    const float* __restrict__ x,
    float* __restrict__ output,
    int rows,
    int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_ptr = x + row * cols;
    __shared__ float smem[BLOCK_SIZE / WARP_SIZE];

    // Pass 1: compute mean
    float local_sum = 0.0f;
    int vec_cols = cols / 4;  // number of float4 elements
    int remainder = cols % 4;

    // Vectorized loads
    const float4* row_ptr4 = reinterpret_cast<const float4*>(row_ptr);
    for (int i = threadIdx.x; i < vec_cols; i += BLOCK_SIZE) {
        float4 v = row_ptr4[i];
        local_sum += v.x + v.y + v.z + v.w;
    }
    // Handle remainder
    int base = vec_cols * 4;
    for (int i = threadIdx.x; i < remainder; i += BLOCK_SIZE) {
        local_sum += row_ptr[base + i];
    }

    float total = block_reduce_sum(local_sum, smem);

    // Broadcast mean to all threads
    __shared__ float mean_shared;
    if (threadIdx.x == 0) {
        mean_shared = total / (float)cols;
    }
    __syncthreads();
    float mean = mean_shared;

    // Pass 2: compute sum of squared deviations
    float local_sq = 0.0f;
    for (int i = threadIdx.x; i < vec_cols; i += BLOCK_SIZE) {
        float4 v = row_ptr4[i];
        float d0 = v.x - mean;
        float d1 = v.y - mean;
        float d2 = v.z - mean;
        float d3 = v.w - mean;
        local_sq += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
    }
    for (int i = threadIdx.x; i < remainder; i += BLOCK_SIZE) {
        float d = row_ptr[base + i] - mean;
        local_sq += d * d;
    }

    float total_sq = block_reduce_sum(local_sq, smem);

    if (threadIdx.x == 0) {
        output[row] = sqrtf(total_sq);
    }
}

// FP16 kernel: one block per row, accumulate in fp32
__global__ void denominator_fp16_kernel(
    const __half* __restrict__ x,
    float* __restrict__ output,
    int rows,
    int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const __half* row_ptr = x + row * cols;
    __shared__ float smem[BLOCK_SIZE / WARP_SIZE];

    // Vectorized loads with half2 (pairs)
    int vec_cols = cols / 2;
    int remainder = cols % 2;
    const __half2* row_ptr2 = reinterpret_cast<const __half2*>(row_ptr);

    // Pass 1: compute mean
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < vec_cols; i += BLOCK_SIZE) {
        __half2 v = row_ptr2[i];
        local_sum += __half2float(v.x) + __half2float(v.y);
    }
    if (remainder && threadIdx.x == 0) {
        local_sum += __half2float(row_ptr[cols - 1]);
    }

    float total = block_reduce_sum(local_sum, smem);

    __shared__ float mean_shared;
    if (threadIdx.x == 0) {
        mean_shared = total / (float)cols;
    }
    __syncthreads();
    float mean = mean_shared;

    // Pass 2: compute sum of squared deviations
    float local_sq = 0.0f;
    for (int i = threadIdx.x; i < vec_cols; i += BLOCK_SIZE) {
        __half2 v = row_ptr2[i];
        float d0 = __half2float(v.x) - mean;
        float d1 = __half2float(v.y) - mean;
        local_sq += d0 * d0 + d1 * d1;
    }
    if (remainder && threadIdx.x == 0) {
        float d = __half2float(row_ptr[cols - 1]) - mean;
        local_sq += d * d;
    }

    float total_sq = block_reduce_sum(local_sq, smem);

    if (threadIdx.x == 0) {
        output[row] = sqrtf(total_sq);
    }
}

// ============================================================================
// V1: Fused denominator + normalize kernel (no streams needed)
// Computes mean, sum-of-squared-deviations, then normalizes raw_output in-place:
//   raw_output[row][c] = raw_output[row][c] / std + b_new[c]
// where std = sqrt(sum_sq / h + eps)
// ============================================================================
__global__ void denominator_normalize_fp32_kernel(
    const float* __restrict__ x,         // [rows, h]
    float* __restrict__ raw_output,      // [rows, out_dim] - modified in-place
    const float* __restrict__ b_new,     // [out_dim]
    int rows,
    int h,
    int out_dim,
    float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* x_row = x + row * h;
    float* out_row = raw_output + row * out_dim;
    __shared__ float smem[BLOCK_SIZE / WARP_SIZE];

    // Pass 1: compute mean
    float local_sum = 0.0f;
    int vec_cols = h / 4;
    int remainder = h % 4;
    const float4* x_row4 = reinterpret_cast<const float4*>(x_row);

    for (int i = threadIdx.x; i < vec_cols; i += BLOCK_SIZE) {
        float4 v = x_row4[i];
        local_sum += v.x + v.y + v.z + v.w;
    }
    int base = vec_cols * 4;
    for (int i = threadIdx.x; i < remainder; i += BLOCK_SIZE) {
        local_sum += x_row[base + i];
    }

    float total = block_reduce_sum(local_sum, smem);

    __shared__ float mean_shared;
    if (threadIdx.x == 0) {
        mean_shared = total / (float)h;
    }
    __syncthreads();
    float mean = mean_shared;

    // Pass 2: compute sum of squared deviations
    float local_sq = 0.0f;
    for (int i = threadIdx.x; i < vec_cols; i += BLOCK_SIZE) {
        float4 v = x_row4[i];
        float d0 = v.x - mean;
        float d1 = v.y - mean;
        float d2 = v.z - mean;
        float d3 = v.w - mean;
        local_sq += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
    }
    for (int i = threadIdx.x; i < remainder; i += BLOCK_SIZE) {
        float d = x_row[base + i] - mean;
        local_sq += d * d;
    }

    float total_sq = block_reduce_sum(local_sq, smem);

    // Broadcast std to all threads via shared memory
    __shared__ float std_shared;
    if (threadIdx.x == 0) {
        std_shared = sqrtf(total_sq / (float)h + eps);
    }
    __syncthreads();
    float std_val = std_shared;
    float inv_std = 1.0f / std_val;

    // Normalize raw_output in-place: out[c] = out[c] / std + b_new[c]
    // Use float4 vectorization for out_dim when possible
    int out_vec = out_dim / 4;
    int out_rem = out_dim % 4;
    float4* out_row4 = reinterpret_cast<float4*>(out_row);
    const float4* b_new4 = reinterpret_cast<const float4*>(b_new);

    for (int i = threadIdx.x; i < out_vec; i += BLOCK_SIZE) {
        float4 o = out_row4[i];
        float4 b = b_new4[i];
        o.x = o.x * inv_std + b.x;
        o.y = o.y * inv_std + b.y;
        o.z = o.z * inv_std + b.z;
        o.w = o.w * inv_std + b.w;
        out_row4[i] = o;
    }
    int out_base = out_vec * 4;
    for (int i = threadIdx.x; i < out_rem; i += BLOCK_SIZE) {
        int c = out_base + i;
        out_row[c] = out_row[c] * inv_std + b_new[c];
    }
}

// ============================================================================
// V1 FP16: Fused denominator + normalize, half input/output, fp32 accumulation
// ============================================================================
__global__ void denominator_normalize_fp16_kernel(
    const __half* __restrict__ x,         // [rows, h]
    __half* __restrict__ raw_output,      // [rows, out_dim] - modified in-place
    const __half* __restrict__ b_new,     // [out_dim]
    int rows,
    int h,
    int out_dim,
    float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const __half* x_row = x + row * h;
    __half* out_row = raw_output + row * out_dim;
    __shared__ float smem[BLOCK_SIZE / WARP_SIZE];

    // Pass 1: compute mean (accumulate in fp32)
    float local_sum = 0.0f;
    int vec_cols = h / 2;
    int remainder = h % 2;
    const __half2* x_row2 = reinterpret_cast<const __half2*>(x_row);

    for (int i = threadIdx.x; i < vec_cols; i += BLOCK_SIZE) {
        __half2 v = x_row2[i];
        local_sum += __half2float(v.x) + __half2float(v.y);
    }
    if (remainder && threadIdx.x == 0) {
        local_sum += __half2float(x_row[h - 1]);
    }

    float total = block_reduce_sum(local_sum, smem);

    __shared__ float mean_shared;
    if (threadIdx.x == 0) {
        mean_shared = total / (float)h;
    }
    __syncthreads();
    float mean = mean_shared;

    // Pass 2: compute sum of squared deviations
    float local_sq = 0.0f;
    for (int i = threadIdx.x; i < vec_cols; i += BLOCK_SIZE) {
        __half2 v = x_row2[i];
        float d0 = __half2float(v.x) - mean;
        float d1 = __half2float(v.y) - mean;
        local_sq += d0 * d0 + d1 * d1;
    }
    if (remainder && threadIdx.x == 0) {
        float d = __half2float(x_row[h - 1]) - mean;
        local_sq += d * d;
    }

    float total_sq = block_reduce_sum(local_sq, smem);

    __shared__ float std_shared;
    if (threadIdx.x == 0) {
        std_shared = sqrtf(total_sq / (float)h + eps);
    }
    __syncthreads();
    float inv_std = 1.0f / std_shared;

    // Normalize raw_output in-place using half2 vectorization
    int out_vec = out_dim / 2;
    int out_rem = out_dim % 2;
    __half2* out_row2 = reinterpret_cast<__half2*>(out_row);
    const __half2* b_new2 = reinterpret_cast<const __half2*>(b_new);

    for (int i = threadIdx.x; i < out_vec; i += BLOCK_SIZE) {
        __half2 o = out_row2[i];
        __half2 b = b_new2[i];
        float o0 = __half2float(o.x) * inv_std + __half2float(b.x);
        float o1 = __half2float(o.y) * inv_std + __half2float(b.y);
        out_row2[i] = __halves2half2(__float2half(o0), __float2half(o1));
    }
    int out_base = out_vec * 2;
    for (int i = threadIdx.x; i < out_rem; i += BLOCK_SIZE) {
        int c = out_base + i;
        float val = __half2float(out_row[c]) * inv_std + __half2float(b_new[c]);
        out_row[c] = __float2half(val);
    }
}

// ============================================================================
// V1 BF16: Fused denominator + normalize, bfloat16 input/output, fp32 accum
// ============================================================================
__global__ void denominator_normalize_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,         // [rows, h]
    __nv_bfloat16* __restrict__ raw_output,      // [rows, out_dim] - modified in-place
    const __nv_bfloat16* __restrict__ b_new,     // [out_dim]
    int rows,
    int h,
    int out_dim,
    float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const __nv_bfloat16* x_row = x + row * h;
    __nv_bfloat16* out_row = raw_output + row * out_dim;
    __shared__ float smem[BLOCK_SIZE / WARP_SIZE];

    // Pass 1: compute mean (accumulate in fp32)
    float local_sum = 0.0f;
    int vec_cols = h / 2;
    int remainder = h % 2;
    const __nv_bfloat162* x_row2 = reinterpret_cast<const __nv_bfloat162*>(x_row);

    for (int i = threadIdx.x; i < vec_cols; i += BLOCK_SIZE) {
        __nv_bfloat162 v = x_row2[i];
        local_sum += __bfloat162float(v.x) + __bfloat162float(v.y);
    }
    if (remainder && threadIdx.x == 0) {
        local_sum += __bfloat162float(x_row[h - 1]);
    }

    float total = block_reduce_sum(local_sum, smem);

    __shared__ float mean_shared;
    if (threadIdx.x == 0) {
        mean_shared = total / (float)h;
    }
    __syncthreads();
    float mean = mean_shared;

    // Pass 2: compute sum of squared deviations
    float local_sq = 0.0f;
    for (int i = threadIdx.x; i < vec_cols; i += BLOCK_SIZE) {
        __nv_bfloat162 v = x_row2[i];
        float d0 = __bfloat162float(v.x) - mean;
        float d1 = __bfloat162float(v.y) - mean;
        local_sq += d0 * d0 + d1 * d1;
    }
    if (remainder && threadIdx.x == 0) {
        float d = __bfloat162float(x_row[h - 1]) - mean;
        local_sq += d * d;
    }

    float total_sq = block_reduce_sum(local_sq, smem);

    __shared__ float std_shared;
    if (threadIdx.x == 0) {
        std_shared = sqrtf(total_sq / (float)h + eps);
    }
    __syncthreads();
    float inv_std = 1.0f / std_shared;

    // Normalize raw_output in-place using bfloat162 vectorization
    int out_vec = out_dim / 2;
    int out_rem = out_dim % 2;
    __nv_bfloat162* out_row2 = reinterpret_cast<__nv_bfloat162*>(out_row);
    const __nv_bfloat162* b_new2 = reinterpret_cast<const __nv_bfloat162*>(b_new);

    for (int i = threadIdx.x; i < out_vec; i += BLOCK_SIZE) {
        __nv_bfloat162 o = out_row2[i];
        __nv_bfloat162 b = b_new2[i];
        float o0 = __bfloat162float(o.x) * inv_std + __bfloat162float(b.x);
        float o1 = __bfloat162float(o.y) * inv_std + __bfloat162float(b.y);
        out_row2[i] = __halves2bfloat162(__float2bfloat16(o0), __float2bfloat16(o1));
    }
    int out_base = out_vec * 2;
    for (int i = threadIdx.x; i < out_rem; i += BLOCK_SIZE) {
        int c = out_base + i;
        float val = __bfloat162float(out_row[c]) * inv_std + __bfloat162float(b_new[c]);
        out_row[c] = __float2bfloat16(val);
    }
}

// ============================================================================
// V2: Welford's single-pass denominator kernel
// Computes mean and sum-of-squared-deviations in one pass, halving memory reads.
// ============================================================================

// Warp-level Welford merge using shuffle
__device__ __forceinline__ void warp_welford_reduce(float& count, float& mean, float& M2) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float o_count = __shfl_down_sync(0xFFFFFFFF, count, offset);
        float o_mean  = __shfl_down_sync(0xFFFFFFFF, mean, offset);
        float o_M2    = __shfl_down_sync(0xFFFFFFFF, M2, offset);
        // Merge: (count, mean, M2) + (o_count, o_mean, o_M2)
        float n = count + o_count;
        if (n > 0.0f) {
            float delta = o_mean - mean;
            float new_mean = mean + delta * o_count / n;
            M2 = M2 + o_M2 + delta * delta * count * o_count / n;
            mean = new_mean;
            count = n;
        }
    }
}

__global__ void denominator_welford_fp32_kernel(
    const float* __restrict__ x,
    float* __restrict__ output,
    int rows,
    int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_ptr = x + row * cols;

    // Each thread maintains a Welford triple
    float count = 0.0f;
    float mean = 0.0f;
    float M2 = 0.0f;

    // Single pass with float4 vectorized loads
    int vec_cols = cols / 4;
    int remainder = cols % 4;
    const float4* row_ptr4 = reinterpret_cast<const float4*>(row_ptr);

    for (int i = threadIdx.x; i < vec_cols; i += BLOCK_SIZE) {
        float4 v = row_ptr4[i];
        // Process each element through Welford update
        float vals[4] = {v.x, v.y, v.z, v.w};
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            count += 1.0f;
            float delta = vals[j] - mean;
            mean += delta / count;
            float delta2 = vals[j] - mean;
            M2 += delta * delta2;
        }
    }
    int base = vec_cols * 4;
    for (int i = threadIdx.x; i < remainder; i += BLOCK_SIZE) {
        float val = row_ptr[base + i];
        count += 1.0f;
        float delta = val - mean;
        mean += delta / count;
        float delta2 = val - mean;
        M2 += delta * delta2;
    }

    // Warp-level reduction
    warp_welford_reduce(count, mean, M2);

    // Inter-warp reduction via shared memory
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int num_warps = BLOCK_SIZE / WARP_SIZE;

    __shared__ float s_count[BLOCK_SIZE / WARP_SIZE];
    __shared__ float s_mean[BLOCK_SIZE / WARP_SIZE];
    __shared__ float s_M2[BLOCK_SIZE / WARP_SIZE];

    if (lane == 0) {
        s_count[warp_id] = count;
        s_mean[warp_id] = mean;
        s_M2[warp_id] = M2;
    }
    __syncthreads();

    // First warp merges all warp results
    if (warp_id == 0) {
        count = (lane < num_warps) ? s_count[lane] : 0.0f;
        mean  = (lane < num_warps) ? s_mean[lane]  : 0.0f;
        M2    = (lane < num_warps) ? s_M2[lane]    : 0.0f;
        warp_welford_reduce(count, mean, M2);
    }

    if (threadIdx.x == 0) {
        // M2 = sum of squared deviations, v = sqrt(M2)
        output[row] = sqrtf(M2);
    }
}

// ============================================================================
// V3: Combined Welford + Fused Normalize + 512 Threads
// Best-of-all-worlds: single-pass, fused normalize, wider blocks.
// ============================================================================
constexpr int BLOCK_SIZE_512 = 512;

// Warp Welford for 512-thread blocks (same logic, separate function for clarity)
__device__ __forceinline__ void warp_welford_reduce_512(float& count, float& mean, float& M2) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float o_count = __shfl_down_sync(0xFFFFFFFF, count, offset);
        float o_mean  = __shfl_down_sync(0xFFFFFFFF, mean, offset);
        float o_M2    = __shfl_down_sync(0xFFFFFFFF, M2, offset);
        float n = count + o_count;
        if (n > 0.0f) {
            float delta = o_mean - mean;
            float new_mean = mean + delta * o_count / n;
            M2 = M2 + o_M2 + delta * delta * count * o_count / n;
            mean = new_mean;
            count = n;
        }
    }
}

__global__ void denominator_normalize_welford_512_fp32_kernel(
    const float* __restrict__ x,         // [rows, h]
    float* __restrict__ raw_output,      // [rows, out_dim] - modified in-place
    const float* __restrict__ b_new,     // [out_dim]
    int rows,
    int h,
    int out_dim,
    float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* x_row = x + row * h;
    float* out_row = raw_output + row * out_dim;

    // Welford single-pass reduction
    float count = 0.0f;
    float mean = 0.0f;
    float M2 = 0.0f;

    int vec_cols = h / 4;
    int remainder = h % 4;
    const float4* x_row4 = reinterpret_cast<const float4*>(x_row);

    for (int i = threadIdx.x; i < vec_cols; i += BLOCK_SIZE_512) {
        float4 v = x_row4[i];
        float vals[4] = {v.x, v.y, v.z, v.w};
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            count += 1.0f;
            float delta = vals[j] - mean;
            mean += delta / count;
            float delta2 = vals[j] - mean;
            M2 += delta * delta2;
        }
    }
    int base = vec_cols * 4;
    for (int i = threadIdx.x; i < remainder; i += BLOCK_SIZE_512) {
        float val = x_row[base + i];
        count += 1.0f;
        float delta = val - mean;
        mean += delta / count;
        float delta2 = val - mean;
        M2 += delta * delta2;
    }

    // Warp-level Welford reduction
    warp_welford_reduce_512(count, mean, M2);

    // Inter-warp reduction via shared memory (16 warps for 512 threads)
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    constexpr int NUM_WARPS_512 = BLOCK_SIZE_512 / WARP_SIZE;  // 16

    __shared__ float s_count[NUM_WARPS_512];
    __shared__ float s_mean[NUM_WARPS_512];
    __shared__ float s_M2[NUM_WARPS_512];

    if (lane == 0) {
        s_count[warp_id] = count;
        s_mean[warp_id] = mean;
        s_M2[warp_id] = M2;
    }
    __syncthreads();

    // First warp merges all 16 warp results
    if (warp_id == 0) {
        count = (lane < NUM_WARPS_512) ? s_count[lane] : 0.0f;
        mean  = (lane < NUM_WARPS_512) ? s_mean[lane]  : 0.0f;
        M2    = (lane < NUM_WARPS_512) ? s_M2[lane]    : 0.0f;
        warp_welford_reduce_512(count, mean, M2);
    }

    // Broadcast std to all threads
    __shared__ float std_shared;
    if (threadIdx.x == 0) {
        std_shared = sqrtf(M2 / (float)h + eps);
    }
    __syncthreads();
    float inv_std = 1.0f / std_shared;

    // Normalize raw_output in-place with float4 vectorization
    int out_vec = out_dim / 4;
    int out_rem = out_dim % 4;
    float4* out_row4 = reinterpret_cast<float4*>(out_row);
    const float4* b_new4 = reinterpret_cast<const float4*>(b_new);

    for (int i = threadIdx.x; i < out_vec; i += BLOCK_SIZE_512) {
        float4 o = out_row4[i];
        float4 b = b_new4[i];
        o.x = o.x * inv_std + b.x;
        o.y = o.y * inv_std + b.y;
        o.z = o.z * inv_std + b.z;
        o.w = o.w * inv_std + b.w;
        out_row4[i] = o;
    }
    int out_base = out_vec * 4;
    for (int i = threadIdx.x; i < out_rem; i += BLOCK_SIZE_512) {
        int c = out_base + i;
        out_row[c] = out_row[c] * inv_std + b_new[c];
    }
}

// ============================================================================
// V3 FP16: Welford + fused normalize + 512 threads, half input/output
// ============================================================================
__global__ void denominator_normalize_welford_512_fp16_kernel(
    const __half* __restrict__ x,         // [rows, h]
    __half* __restrict__ raw_output,      // [rows, out_dim]
    const __half* __restrict__ b_new,     // [out_dim]
    int rows,
    int h,
    int out_dim,
    float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const __half* x_row = x + row * h;
    __half* out_row = raw_output + row * out_dim;

    // Welford single-pass (fp32 accumulation)
    float count = 0.0f;
    float mean = 0.0f;
    float M2 = 0.0f;

    int vec_cols = h / 2;
    int remainder = h % 2;
    const __half2* x_row2 = reinterpret_cast<const __half2*>(x_row);

    for (int i = threadIdx.x; i < vec_cols; i += BLOCK_SIZE_512) {
        __half2 v = x_row2[i];
        float vals[2] = {__half2float(v.x), __half2float(v.y)};
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            count += 1.0f;
            float delta = vals[j] - mean;
            mean += delta / count;
            float delta2 = vals[j] - mean;
            M2 += delta * delta2;
        }
    }
    if (remainder && threadIdx.x == 0) {
        float val = __half2float(x_row[h - 1]);
        count += 1.0f;
        float delta = val - mean;
        mean += delta / count;
        float delta2 = val - mean;
        M2 += delta * delta2;
    }

    warp_welford_reduce_512(count, mean, M2);

    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    constexpr int NUM_WARPS_512 = BLOCK_SIZE_512 / WARP_SIZE;

    __shared__ float s_count[NUM_WARPS_512];
    __shared__ float s_mean[NUM_WARPS_512];
    __shared__ float s_M2[NUM_WARPS_512];

    if (lane == 0) {
        s_count[warp_id] = count;
        s_mean[warp_id] = mean;
        s_M2[warp_id] = M2;
    }
    __syncthreads();

    if (warp_id == 0) {
        count = (lane < NUM_WARPS_512) ? s_count[lane] : 0.0f;
        mean  = (lane < NUM_WARPS_512) ? s_mean[lane]  : 0.0f;
        M2    = (lane < NUM_WARPS_512) ? s_M2[lane]    : 0.0f;
        warp_welford_reduce_512(count, mean, M2);
    }

    __shared__ float std_shared;
    if (threadIdx.x == 0) {
        std_shared = sqrtf(M2 / (float)h + eps);
    }
    __syncthreads();
    float inv_std = 1.0f / std_shared;

    // Normalize with half2 vectorization
    int out_vec = out_dim / 2;
    int out_rem = out_dim % 2;
    __half2* out_row2 = reinterpret_cast<__half2*>(out_row);
    const __half2* b_new2 = reinterpret_cast<const __half2*>(b_new);

    for (int i = threadIdx.x; i < out_vec; i += BLOCK_SIZE_512) {
        __half2 o = out_row2[i];
        __half2 b = b_new2[i];
        float o0 = __half2float(o.x) * inv_std + __half2float(b.x);
        float o1 = __half2float(o.y) * inv_std + __half2float(b.y);
        out_row2[i] = __halves2half2(__float2half(o0), __float2half(o1));
    }
    int out_base = out_vec * 2;
    for (int i = threadIdx.x; i < out_rem; i += BLOCK_SIZE_512) {
        int c = out_base + i;
        float val = __half2float(out_row[c]) * inv_std + __half2float(b_new[c]);
        out_row[c] = __float2half(val);
    }
}

// ============================================================================
// V3 BF16: Welford + fused normalize + 512 threads, bfloat16 input/output
// ============================================================================
__global__ void denominator_normalize_welford_512_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,         // [rows, h]
    __nv_bfloat16* __restrict__ raw_output,      // [rows, out_dim]
    const __nv_bfloat16* __restrict__ b_new,     // [out_dim]
    int rows,
    int h,
    int out_dim,
    float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const __nv_bfloat16* x_row = x + row * h;
    __nv_bfloat16* out_row = raw_output + row * out_dim;

    // Welford single-pass (fp32 accumulation)
    float count = 0.0f;
    float mean = 0.0f;
    float M2 = 0.0f;

    int vec_cols = h / 2;
    int remainder = h % 2;
    const __nv_bfloat162* x_row2 = reinterpret_cast<const __nv_bfloat162*>(x_row);

    for (int i = threadIdx.x; i < vec_cols; i += BLOCK_SIZE_512) {
        __nv_bfloat162 v = x_row2[i];
        float vals[2] = {__bfloat162float(v.x), __bfloat162float(v.y)};
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            count += 1.0f;
            float delta = vals[j] - mean;
            mean += delta / count;
            float delta2 = vals[j] - mean;
            M2 += delta * delta2;
        }
    }
    if (remainder && threadIdx.x == 0) {
        float val = __bfloat162float(x_row[h - 1]);
        count += 1.0f;
        float delta = val - mean;
        mean += delta / count;
        float delta2 = val - mean;
        M2 += delta * delta2;
    }

    warp_welford_reduce_512(count, mean, M2);

    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    constexpr int NUM_WARPS_512 = BLOCK_SIZE_512 / WARP_SIZE;

    __shared__ float s_count[NUM_WARPS_512];
    __shared__ float s_mean[NUM_WARPS_512];
    __shared__ float s_M2[NUM_WARPS_512];

    if (lane == 0) {
        s_count[warp_id] = count;
        s_mean[warp_id] = mean;
        s_M2[warp_id] = M2;
    }
    __syncthreads();

    if (warp_id == 0) {
        count = (lane < NUM_WARPS_512) ? s_count[lane] : 0.0f;
        mean  = (lane < NUM_WARPS_512) ? s_mean[lane]  : 0.0f;
        M2    = (lane < NUM_WARPS_512) ? s_M2[lane]    : 0.0f;
        warp_welford_reduce_512(count, mean, M2);
    }

    __shared__ float std_shared;
    if (threadIdx.x == 0) {
        std_shared = sqrtf(M2 / (float)h + eps);
    }
    __syncthreads();
    float inv_std = 1.0f / std_shared;

    // Normalize with bfloat162 vectorization
    int out_vec = out_dim / 2;
    int out_rem = out_dim % 2;
    __nv_bfloat162* out_row2 = reinterpret_cast<__nv_bfloat162*>(out_row);
    const __nv_bfloat162* b_new2 = reinterpret_cast<const __nv_bfloat162*>(b_new);

    for (int i = threadIdx.x; i < out_vec; i += BLOCK_SIZE_512) {
        __nv_bfloat162 o = out_row2[i];
        __nv_bfloat162 b = b_new2[i];
        float o0 = __bfloat162float(o.x) * inv_std + __bfloat162float(b.x);
        float o1 = __bfloat162float(o.y) * inv_std + __bfloat162float(b.y);
        out_row2[i] = __halves2bfloat162(__float2bfloat16(o0), __float2bfloat16(o1));
    }
    int out_base = out_vec * 2;
    for (int i = threadIdx.x; i < out_rem; i += BLOCK_SIZE_512) {
        int c = out_base + i;
        float val = __bfloat162float(out_row[c]) * inv_std + __bfloat162float(b_new[c]);
        out_row[c] = __float2bfloat16(val);
    }
}

// ============================================================================
// RMSNorm kernels
// RMSNorm: output = x / rms(x) * gamma, rms(x) = sqrt(mean(x^2) + eps)
// No mean subtraction — simpler than LayerNorm. Single pass suffices.
// The weight absorption is done on CPU: W_new = W * gamma
// So at runtime we compute: raw_output / rms(x) + b_new
// ============================================================================

// V1 RMSNorm FP32: single pass, 256 threads
__global__ void rmsnorm_normalize_fp32_kernel(
    const float* __restrict__ x,         // [rows, h]
    float* __restrict__ raw_output,      // [rows, out_dim] - modified in-place
    const float* __restrict__ b_new,     // [out_dim]
    int rows,
    int h,
    int out_dim,
    float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* x_row = x + row * h;
    float* out_row = raw_output + row * out_dim;
    __shared__ float smem[BLOCK_SIZE / WARP_SIZE];

    // Single pass: sum of squares
    float local_sq = 0.0f;
    int vec_cols = h / 4;
    int remainder = h % 4;
    const float4* x_row4 = reinterpret_cast<const float4*>(x_row);

    for (int i = threadIdx.x; i < vec_cols; i += BLOCK_SIZE) {
        float4 v = x_row4[i];
        local_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    int base = vec_cols * 4;
    for (int i = threadIdx.x; i < remainder; i += BLOCK_SIZE) {
        float val = x_row[base + i];
        local_sq += val * val;
    }

    float total_sq = block_reduce_sum(local_sq, smem);

    // rms = sqrt(mean(x^2) + eps)
    __shared__ float rms_shared;
    if (threadIdx.x == 0) {
        rms_shared = sqrtf(total_sq / (float)h + eps);
    }
    __syncthreads();
    float inv_rms = 1.0f / rms_shared;

    // Normalize raw_output in-place
    int out_vec = out_dim / 4;
    int out_rem = out_dim % 4;
    float4* out_row4 = reinterpret_cast<float4*>(out_row);
    const float4* b_new4 = reinterpret_cast<const float4*>(b_new);

    for (int i = threadIdx.x; i < out_vec; i += BLOCK_SIZE) {
        float4 o = out_row4[i];
        float4 b = b_new4[i];
        o.x = o.x * inv_rms + b.x;
        o.y = o.y * inv_rms + b.y;
        o.z = o.z * inv_rms + b.z;
        o.w = o.w * inv_rms + b.w;
        out_row4[i] = o;
    }
    int out_base = out_vec * 4;
    for (int i = threadIdx.x; i < out_rem; i += BLOCK_SIZE) {
        int c = out_base + i;
        out_row[c] = out_row[c] * inv_rms + b_new[c];
    }
}

// V1 RMSNorm FP16
__global__ void rmsnorm_normalize_fp16_kernel(
    const __half* __restrict__ x,
    __half* __restrict__ raw_output,
    const __half* __restrict__ b_new,
    int rows,
    int h,
    int out_dim,
    float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const __half* x_row = x + row * h;
    __half* out_row = raw_output + row * out_dim;
    __shared__ float smem[BLOCK_SIZE / WARP_SIZE];

    float local_sq = 0.0f;
    int vec_cols = h / 2;
    int remainder = h % 2;
    const __half2* x_row2 = reinterpret_cast<const __half2*>(x_row);

    for (int i = threadIdx.x; i < vec_cols; i += BLOCK_SIZE) {
        __half2 v = x_row2[i];
        float v0 = __half2float(v.x), v1 = __half2float(v.y);
        local_sq += v0 * v0 + v1 * v1;
    }
    if (remainder && threadIdx.x == 0) {
        float v = __half2float(x_row[h - 1]);
        local_sq += v * v;
    }

    float total_sq = block_reduce_sum(local_sq, smem);

    __shared__ float rms_shared;
    if (threadIdx.x == 0) {
        rms_shared = sqrtf(total_sq / (float)h + eps);
    }
    __syncthreads();
    float inv_rms = 1.0f / rms_shared;

    int out_vec = out_dim / 2;
    int out_rem = out_dim % 2;
    __half2* out_row2 = reinterpret_cast<__half2*>(out_row);
    const __half2* b_new2 = reinterpret_cast<const __half2*>(b_new);

    for (int i = threadIdx.x; i < out_vec; i += BLOCK_SIZE) {
        __half2 o = out_row2[i];
        __half2 b = b_new2[i];
        float o0 = __half2float(o.x) * inv_rms + __half2float(b.x);
        float o1 = __half2float(o.y) * inv_rms + __half2float(b.y);
        out_row2[i] = __halves2half2(__float2half(o0), __float2half(o1));
    }
    int out_base = out_vec * 2;
    for (int i = threadIdx.x; i < out_rem; i += BLOCK_SIZE) {
        int c = out_base + i;
        float val = __half2float(out_row[c]) * inv_rms + __half2float(b_new[c]);
        out_row[c] = __float2half(val);
    }
}

// V1 RMSNorm BF16
__global__ void rmsnorm_normalize_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ raw_output,
    const __nv_bfloat16* __restrict__ b_new,
    int rows,
    int h,
    int out_dim,
    float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const __nv_bfloat16* x_row = x + row * h;
    __nv_bfloat16* out_row = raw_output + row * out_dim;
    __shared__ float smem[BLOCK_SIZE / WARP_SIZE];

    float local_sq = 0.0f;
    int vec_cols = h / 2;
    int remainder = h % 2;
    const __nv_bfloat162* x_row2 = reinterpret_cast<const __nv_bfloat162*>(x_row);

    for (int i = threadIdx.x; i < vec_cols; i += BLOCK_SIZE) {
        __nv_bfloat162 v = x_row2[i];
        float v0 = __bfloat162float(v.x), v1 = __bfloat162float(v.y);
        local_sq += v0 * v0 + v1 * v1;
    }
    if (remainder && threadIdx.x == 0) {
        float v = __bfloat162float(x_row[h - 1]);
        local_sq += v * v;
    }

    float total_sq = block_reduce_sum(local_sq, smem);

    __shared__ float rms_shared;
    if (threadIdx.x == 0) {
        rms_shared = sqrtf(total_sq / (float)h + eps);
    }
    __syncthreads();
    float inv_rms = 1.0f / rms_shared;

    int out_vec = out_dim / 2;
    int out_rem = out_dim % 2;
    __nv_bfloat162* out_row2 = reinterpret_cast<__nv_bfloat162*>(out_row);
    const __nv_bfloat162* b_new2 = reinterpret_cast<const __nv_bfloat162*>(b_new);

    for (int i = threadIdx.x; i < out_vec; i += BLOCK_SIZE) {
        __nv_bfloat162 o = out_row2[i];
        __nv_bfloat162 b = b_new2[i];
        float o0 = __bfloat162float(o.x) * inv_rms + __bfloat162float(b.x);
        float o1 = __bfloat162float(o.y) * inv_rms + __bfloat162float(b.y);
        out_row2[i] = __halves2bfloat162(__float2bfloat16(o0), __float2bfloat16(o1));
    }
    int out_base = out_vec * 2;
    for (int i = threadIdx.x; i < out_rem; i += BLOCK_SIZE) {
        int c = out_base + i;
        float val = __bfloat162float(out_row[c]) * inv_rms + __bfloat162float(b_new[c]);
        out_row[c] = __float2bfloat16(val);
    }
}

// V3 RMSNorm FP32: 512 threads
__global__ void rmsnorm_normalize_512_fp32_kernel(
    const float* __restrict__ x,
    float* __restrict__ raw_output,
    const float* __restrict__ b_new,
    int rows,
    int h,
    int out_dim,
    float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* x_row = x + row * h;
    float* out_row = raw_output + row * out_dim;
    __shared__ float smem[BLOCK_SIZE_512 / WARP_SIZE];

    float local_sq = 0.0f;
    int vec_cols = h / 4;
    int remainder = h % 4;
    const float4* x_row4 = reinterpret_cast<const float4*>(x_row);

    for (int i = threadIdx.x; i < vec_cols; i += BLOCK_SIZE_512) {
        float4 v = x_row4[i];
        local_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    int base = vec_cols * 4;
    for (int i = threadIdx.x; i < remainder; i += BLOCK_SIZE_512) {
        float val = x_row[base + i];
        local_sq += val * val;
    }

    // Block reduce with 512-thread shared memory
    float total_sq;
    {
        int lane = threadIdx.x % WARP_SIZE;
        int warp_id = threadIdx.x / WARP_SIZE;
        local_sq = warp_reduce_sum(local_sq);
        if (lane == 0) smem[warp_id] = local_sq;
        __syncthreads();
        constexpr int NUM_WARPS = BLOCK_SIZE_512 / WARP_SIZE;
        if (warp_id == 0) {
            local_sq = (lane < NUM_WARPS) ? smem[lane] : 0.0f;
            local_sq = warp_reduce_sum(local_sq);
        }
        total_sq = local_sq;
    }

    __shared__ float rms_shared;
    if (threadIdx.x == 0) {
        rms_shared = sqrtf(total_sq / (float)h + eps);
    }
    __syncthreads();
    float inv_rms = 1.0f / rms_shared;

    int out_vec = out_dim / 4;
    int out_rem = out_dim % 4;
    float4* out_row4 = reinterpret_cast<float4*>(out_row);
    const float4* b_new4 = reinterpret_cast<const float4*>(b_new);

    for (int i = threadIdx.x; i < out_vec; i += BLOCK_SIZE_512) {
        float4 o = out_row4[i];
        float4 b = b_new4[i];
        o.x = o.x * inv_rms + b.x;
        o.y = o.y * inv_rms + b.y;
        o.z = o.z * inv_rms + b.z;
        o.w = o.w * inv_rms + b.w;
        out_row4[i] = o;
    }
    int out_base = out_vec * 4;
    for (int i = threadIdx.x; i < out_rem; i += BLOCK_SIZE_512) {
        int c = out_base + i;
        out_row[c] = out_row[c] * inv_rms + b_new[c];
    }
}

// V3 RMSNorm FP16: 512 threads
__global__ void rmsnorm_normalize_512_fp16_kernel(
    const __half* __restrict__ x,
    __half* __restrict__ raw_output,
    const __half* __restrict__ b_new,
    int rows,
    int h,
    int out_dim,
    float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const __half* x_row = x + row * h;
    __half* out_row = raw_output + row * out_dim;
    __shared__ float smem[BLOCK_SIZE_512 / WARP_SIZE];

    float local_sq = 0.0f;
    int vec_cols = h / 2;
    int remainder = h % 2;
    const __half2* x_row2 = reinterpret_cast<const __half2*>(x_row);

    for (int i = threadIdx.x; i < vec_cols; i += BLOCK_SIZE_512) {
        __half2 v = x_row2[i];
        float v0 = __half2float(v.x), v1 = __half2float(v.y);
        local_sq += v0 * v0 + v1 * v1;
    }
    if (remainder && threadIdx.x == 0) {
        float v = __half2float(x_row[h - 1]);
        local_sq += v * v;
    }

    float total_sq;
    {
        int lane = threadIdx.x % WARP_SIZE;
        int warp_id = threadIdx.x / WARP_SIZE;
        local_sq = warp_reduce_sum(local_sq);
        if (lane == 0) smem[warp_id] = local_sq;
        __syncthreads();
        constexpr int NUM_WARPS = BLOCK_SIZE_512 / WARP_SIZE;
        if (warp_id == 0) {
            local_sq = (lane < NUM_WARPS) ? smem[lane] : 0.0f;
            local_sq = warp_reduce_sum(local_sq);
        }
        total_sq = local_sq;
    }

    __shared__ float rms_shared;
    if (threadIdx.x == 0) {
        rms_shared = sqrtf(total_sq / (float)h + eps);
    }
    __syncthreads();
    float inv_rms = 1.0f / rms_shared;

    int out_vec = out_dim / 2;
    int out_rem = out_dim % 2;
    __half2* out_row2 = reinterpret_cast<__half2*>(out_row);
    const __half2* b_new2 = reinterpret_cast<const __half2*>(b_new);

    for (int i = threadIdx.x; i < out_vec; i += BLOCK_SIZE_512) {
        __half2 o = out_row2[i];
        __half2 b = b_new2[i];
        float o0 = __half2float(o.x) * inv_rms + __half2float(b.x);
        float o1 = __half2float(o.y) * inv_rms + __half2float(b.y);
        out_row2[i] = __halves2half2(__float2half(o0), __float2half(o1));
    }
    int out_base = out_vec * 2;
    for (int i = threadIdx.x; i < out_rem; i += BLOCK_SIZE_512) {
        int c = out_base + i;
        float val = __half2float(out_row[c]) * inv_rms + __half2float(b_new[c]);
        out_row[c] = __float2half(val);
    }
}

// V3 RMSNorm BF16: 512 threads
__global__ void rmsnorm_normalize_512_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ raw_output,
    const __nv_bfloat16* __restrict__ b_new,
    int rows,
    int h,
    int out_dim,
    float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const __nv_bfloat16* x_row = x + row * h;
    __nv_bfloat16* out_row = raw_output + row * out_dim;
    __shared__ float smem[BLOCK_SIZE_512 / WARP_SIZE];

    float local_sq = 0.0f;
    int vec_cols = h / 2;
    int remainder = h % 2;
    const __nv_bfloat162* x_row2 = reinterpret_cast<const __nv_bfloat162*>(x_row);

    for (int i = threadIdx.x; i < vec_cols; i += BLOCK_SIZE_512) {
        __nv_bfloat162 v = x_row2[i];
        float v0 = __bfloat162float(v.x), v1 = __bfloat162float(v.y);
        local_sq += v0 * v0 + v1 * v1;
    }
    if (remainder && threadIdx.x == 0) {
        float v = __bfloat162float(x_row[h - 1]);
        local_sq += v * v;
    }

    float total_sq;
    {
        int lane = threadIdx.x % WARP_SIZE;
        int warp_id = threadIdx.x / WARP_SIZE;
        local_sq = warp_reduce_sum(local_sq);
        if (lane == 0) smem[warp_id] = local_sq;
        __syncthreads();
        constexpr int NUM_WARPS = BLOCK_SIZE_512 / WARP_SIZE;
        if (warp_id == 0) {
            local_sq = (lane < NUM_WARPS) ? smem[lane] : 0.0f;
            local_sq = warp_reduce_sum(local_sq);
        }
        total_sq = local_sq;
    }

    __shared__ float rms_shared;
    if (threadIdx.x == 0) {
        rms_shared = sqrtf(total_sq / (float)h + eps);
    }
    __syncthreads();
    float inv_rms = 1.0f / rms_shared;

    int out_vec = out_dim / 2;
    int out_rem = out_dim % 2;
    __nv_bfloat162* out_row2 = reinterpret_cast<__nv_bfloat162*>(out_row);
    const __nv_bfloat162* b_new2 = reinterpret_cast<const __nv_bfloat162*>(b_new);

    for (int i = threadIdx.x; i < out_vec; i += BLOCK_SIZE_512) {
        __nv_bfloat162 o = out_row2[i];
        __nv_bfloat162 b = b_new2[i];
        float o0 = __bfloat162float(o.x) * inv_rms + __bfloat162float(b.x);
        float o1 = __bfloat162float(o.y) * inv_rms + __bfloat162float(b.y);
        out_row2[i] = __halves2bfloat162(__float2bfloat16(o0), __float2bfloat16(o1));
    }
    int out_base = out_vec * 2;
    for (int i = threadIdx.x; i < out_rem; i += BLOCK_SIZE_512) {
        int c = out_base + i;
        float val = __bfloat162float(out_row[c]) * inv_rms + __bfloat162float(b_new[c]);
        out_row[c] = __float2bfloat16(val);
    }
}

// ============================================================================
// RMSNorm + SwiGLU fused kernels
// Phase 1: RMS reduction over x (sum of squares)
// Phase 2: SwiGLU epilogue — read gate/up halves from raw_output, normalize,
//          apply SiLU(gate) * up, write to separate output buffer.
//
// raw_output: [rows, 2*intermediate]  (gate | up concatenated)
// output:     [rows, intermediate]    (SiLU(norm(gate)) * norm(up))
// ============================================================================

// V1 RMSNorm+SwiGLU FP32: 256 threads
__global__ void rmsnorm_swiglu_fp32_kernel(
    const float* __restrict__ x,            // [rows, h]
    const float* __restrict__ raw_output,   // [rows, 2*intermediate] READ-ONLY
    float* __restrict__ output,             // [rows, intermediate]
    const float* __restrict__ b_new,        // [2*intermediate]
    int rows,
    int h,
    int intermediate,
    float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* x_row = x + row * h;
    __shared__ float smem[BLOCK_SIZE / WARP_SIZE];

    // Phase 1: sum of squares for RMS
    float local_sq = 0.0f;
    int vec_cols = h / 4;
    int remainder = h % 4;
    const float4* x_row4 = reinterpret_cast<const float4*>(x_row);

    for (int i = threadIdx.x; i < vec_cols; i += BLOCK_SIZE) {
        float4 v = x_row4[i];
        local_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    int base = vec_cols * 4;
    for (int i = threadIdx.x; i < remainder; i += BLOCK_SIZE) {
        float val = x_row[base + i];
        local_sq += val * val;
    }

    float total_sq = block_reduce_sum(local_sq, smem);

    __shared__ float rms_shared;
    if (threadIdx.x == 0) {
        rms_shared = sqrtf(total_sq / (float)h + eps);
    }
    __syncthreads();
    float inv_rms = 1.0f / rms_shared;

    // Phase 2: SwiGLU epilogue
    const float* gate_row = raw_output + row * 2 * intermediate;
    const float* up_row   = gate_row + intermediate;
    float* out_row        = output + row * intermediate;
    const float* b_gate   = b_new;
    const float* b_up     = b_new + intermediate;

    int out_vec = intermediate / 4;
    int out_rem = intermediate % 4;

    for (int i = threadIdx.x; i < out_vec; i += BLOCK_SIZE) {
        float4 g = reinterpret_cast<const float4*>(gate_row)[i];
        float4 u = reinterpret_cast<const float4*>(up_row)[i];
        float4 bg = reinterpret_cast<const float4*>(b_gate)[i];
        float4 bu = reinterpret_cast<const float4*>(b_up)[i];
        float4 result;
        // x: normalize + SiLU + multiply
        float gv = g.x * inv_rms + bg.x;
        float uv = u.x * inv_rms + bu.x;
        result.x = (gv / (1.0f + expf(-gv))) * uv;
        // y
        gv = g.y * inv_rms + bg.y;
        uv = u.y * inv_rms + bu.y;
        result.y = (gv / (1.0f + expf(-gv))) * uv;
        // z
        gv = g.z * inv_rms + bg.z;
        uv = u.z * inv_rms + bu.z;
        result.z = (gv / (1.0f + expf(-gv))) * uv;
        // w
        gv = g.w * inv_rms + bg.w;
        uv = u.w * inv_rms + bu.w;
        result.w = (gv / (1.0f + expf(-gv))) * uv;
        reinterpret_cast<float4*>(out_row)[i] = result;
    }
    int out_base = out_vec * 4;
    for (int i = threadIdx.x; i < out_rem; i += BLOCK_SIZE) {
        int c = out_base + i;
        float gv = gate_row[c] * inv_rms + b_gate[c];
        float uv = up_row[c] * inv_rms + b_up[c];
        out_row[c] = (gv / (1.0f + expf(-gv))) * uv;
    }
}

// V1 RMSNorm+SwiGLU FP16: 256 threads
__global__ void rmsnorm_swiglu_fp16_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ raw_output,
    __half* __restrict__ output,
    const __half* __restrict__ b_new,
    int rows,
    int h,
    int intermediate,
    float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const __half* x_row = x + row * h;
    __shared__ float smem[BLOCK_SIZE / WARP_SIZE];

    float local_sq = 0.0f;
    int vec_cols = h / 2;
    int remainder = h % 2;
    const __half2* x_row2 = reinterpret_cast<const __half2*>(x_row);

    for (int i = threadIdx.x; i < vec_cols; i += BLOCK_SIZE) {
        __half2 v = x_row2[i];
        float v0 = __half2float(v.x), v1 = __half2float(v.y);
        local_sq += v0 * v0 + v1 * v1;
    }
    if (remainder && threadIdx.x == 0) {
        float v = __half2float(x_row[h - 1]);
        local_sq += v * v;
    }

    float total_sq = block_reduce_sum(local_sq, smem);

    __shared__ float rms_shared;
    if (threadIdx.x == 0) {
        rms_shared = sqrtf(total_sq / (float)h + eps);
    }
    __syncthreads();
    float inv_rms = 1.0f / rms_shared;

    const __half* gate_row = raw_output + row * 2 * intermediate;
    const __half* up_row   = gate_row + intermediate;
    __half* out_row        = output + row * intermediate;
    const __half* b_gate   = b_new;
    const __half* b_up     = b_new + intermediate;

    int out_vec = intermediate / 2;
    int out_rem = intermediate % 2;

    for (int i = threadIdx.x; i < out_vec; i += BLOCK_SIZE) {
        __half2 g = reinterpret_cast<const __half2*>(gate_row)[i];
        __half2 u = reinterpret_cast<const __half2*>(up_row)[i];
        __half2 bg = reinterpret_cast<const __half2*>(b_gate)[i];
        __half2 bu = reinterpret_cast<const __half2*>(b_up)[i];
        float gv0 = __half2float(g.x) * inv_rms + __half2float(bg.x);
        float uv0 = __half2float(u.x) * inv_rms + __half2float(bu.x);
        float r0 = (gv0 / (1.0f + expf(-gv0))) * uv0;
        float gv1 = __half2float(g.y) * inv_rms + __half2float(bg.y);
        float uv1 = __half2float(u.y) * inv_rms + __half2float(bu.y);
        float r1 = (gv1 / (1.0f + expf(-gv1))) * uv1;
        reinterpret_cast<__half2*>(out_row)[i] = __halves2half2(__float2half(r0), __float2half(r1));
    }
    int out_base = out_vec * 2;
    for (int i = threadIdx.x; i < out_rem; i += BLOCK_SIZE) {
        int c = out_base + i;
        float gv = __half2float(gate_row[c]) * inv_rms + __half2float(b_gate[c]);
        float uv = __half2float(up_row[c]) * inv_rms + __half2float(b_up[c]);
        out_row[c] = __float2half((gv / (1.0f + expf(-gv))) * uv);
    }
}

// V1 RMSNorm+SwiGLU BF16: 256 threads
__global__ void rmsnorm_swiglu_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ raw_output,
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ b_new,
    int rows,
    int h,
    int intermediate,
    float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const __nv_bfloat16* x_row = x + row * h;
    __shared__ float smem[BLOCK_SIZE / WARP_SIZE];

    float local_sq = 0.0f;
    int vec_cols = h / 2;
    int remainder = h % 2;
    const __nv_bfloat162* x_row2 = reinterpret_cast<const __nv_bfloat162*>(x_row);

    for (int i = threadIdx.x; i < vec_cols; i += BLOCK_SIZE) {
        __nv_bfloat162 v = x_row2[i];
        float v0 = __bfloat162float(v.x), v1 = __bfloat162float(v.y);
        local_sq += v0 * v0 + v1 * v1;
    }
    if (remainder && threadIdx.x == 0) {
        float v = __bfloat162float(x_row[h - 1]);
        local_sq += v * v;
    }

    float total_sq = block_reduce_sum(local_sq, smem);

    __shared__ float rms_shared;
    if (threadIdx.x == 0) {
        rms_shared = sqrtf(total_sq / (float)h + eps);
    }
    __syncthreads();
    float inv_rms = 1.0f / rms_shared;

    const __nv_bfloat16* gate_row = raw_output + row * 2 * intermediate;
    const __nv_bfloat16* up_row   = gate_row + intermediate;
    __nv_bfloat16* out_row        = output + row * intermediate;
    const __nv_bfloat16* b_gate   = b_new;
    const __nv_bfloat16* b_up     = b_new + intermediate;

    int out_vec = intermediate / 2;
    int out_rem = intermediate % 2;

    for (int i = threadIdx.x; i < out_vec; i += BLOCK_SIZE) {
        __nv_bfloat162 g = reinterpret_cast<const __nv_bfloat162*>(gate_row)[i];
        __nv_bfloat162 u = reinterpret_cast<const __nv_bfloat162*>(up_row)[i];
        __nv_bfloat162 bg = reinterpret_cast<const __nv_bfloat162*>(b_gate)[i];
        __nv_bfloat162 bu = reinterpret_cast<const __nv_bfloat162*>(b_up)[i];
        float gv0 = __bfloat162float(g.x) * inv_rms + __bfloat162float(bg.x);
        float uv0 = __bfloat162float(u.x) * inv_rms + __bfloat162float(bu.x);
        float r0 = (gv0 / (1.0f + expf(-gv0))) * uv0;
        float gv1 = __bfloat162float(g.y) * inv_rms + __bfloat162float(bg.y);
        float uv1 = __bfloat162float(u.y) * inv_rms + __bfloat162float(bu.y);
        float r1 = (gv1 / (1.0f + expf(-gv1))) * uv1;
        reinterpret_cast<__nv_bfloat162*>(out_row)[i] = __halves2bfloat162(__float2bfloat16(r0), __float2bfloat16(r1));
    }
    int out_base = out_vec * 2;
    for (int i = threadIdx.x; i < out_rem; i += BLOCK_SIZE) {
        int c = out_base + i;
        float gv = __bfloat162float(gate_row[c]) * inv_rms + __bfloat162float(b_gate[c]);
        float uv = __bfloat162float(up_row[c]) * inv_rms + __bfloat162float(b_up[c]);
        out_row[c] = __float2bfloat16((gv / (1.0f + expf(-gv))) * uv);
    }
}

// V3 RMSNorm+SwiGLU FP32: 512 threads
__global__ void rmsnorm_swiglu_512_fp32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ raw_output,
    float* __restrict__ output,
    const float* __restrict__ b_new,
    int rows,
    int h,
    int intermediate,
    float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* x_row = x + row * h;
    __shared__ float smem[BLOCK_SIZE_512 / WARP_SIZE];

    float local_sq = 0.0f;
    int vec_cols = h / 4;
    int remainder = h % 4;
    const float4* x_row4 = reinterpret_cast<const float4*>(x_row);

    for (int i = threadIdx.x; i < vec_cols; i += BLOCK_SIZE_512) {
        float4 v = x_row4[i];
        local_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    int base = vec_cols * 4;
    for (int i = threadIdx.x; i < remainder; i += BLOCK_SIZE_512) {
        float val = x_row[base + i];
        local_sq += val * val;
    }

    // Block reduce with 512-thread shared memory
    float total_sq;
    {
        int lane = threadIdx.x % WARP_SIZE;
        int warp_id = threadIdx.x / WARP_SIZE;
        local_sq = warp_reduce_sum(local_sq);
        if (lane == 0) smem[warp_id] = local_sq;
        __syncthreads();
        constexpr int NUM_WARPS = BLOCK_SIZE_512 / WARP_SIZE;
        if (warp_id == 0) {
            local_sq = (lane < NUM_WARPS) ? smem[lane] : 0.0f;
            local_sq = warp_reduce_sum(local_sq);
        }
        total_sq = local_sq;
    }

    __shared__ float rms_shared;
    if (threadIdx.x == 0) {
        rms_shared = sqrtf(total_sq / (float)h + eps);
    }
    __syncthreads();
    float inv_rms = 1.0f / rms_shared;

    const float* gate_row = raw_output + row * 2 * intermediate;
    const float* up_row   = gate_row + intermediate;
    float* out_row        = output + row * intermediate;
    const float* b_gate   = b_new;
    const float* b_up     = b_new + intermediate;

    int out_vec = intermediate / 4;
    int out_rem = intermediate % 4;

    for (int i = threadIdx.x; i < out_vec; i += BLOCK_SIZE_512) {
        float4 g = reinterpret_cast<const float4*>(gate_row)[i];
        float4 u = reinterpret_cast<const float4*>(up_row)[i];
        float4 bg = reinterpret_cast<const float4*>(b_gate)[i];
        float4 bu = reinterpret_cast<const float4*>(b_up)[i];
        float4 result;
        float gv = g.x * inv_rms + bg.x;
        float uv = u.x * inv_rms + bu.x;
        result.x = (gv / (1.0f + expf(-gv))) * uv;
        gv = g.y * inv_rms + bg.y;
        uv = u.y * inv_rms + bu.y;
        result.y = (gv / (1.0f + expf(-gv))) * uv;
        gv = g.z * inv_rms + bg.z;
        uv = u.z * inv_rms + bu.z;
        result.z = (gv / (1.0f + expf(-gv))) * uv;
        gv = g.w * inv_rms + bg.w;
        uv = u.w * inv_rms + bu.w;
        result.w = (gv / (1.0f + expf(-gv))) * uv;
        reinterpret_cast<float4*>(out_row)[i] = result;
    }
    int out_base = out_vec * 4;
    for (int i = threadIdx.x; i < out_rem; i += BLOCK_SIZE_512) {
        int c = out_base + i;
        float gv = gate_row[c] * inv_rms + b_gate[c];
        float uv = up_row[c] * inv_rms + b_up[c];
        out_row[c] = (gv / (1.0f + expf(-gv))) * uv;
    }
}

// V3 RMSNorm+SwiGLU FP16: 512 threads
__global__ void rmsnorm_swiglu_512_fp16_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ raw_output,
    __half* __restrict__ output,
    const __half* __restrict__ b_new,
    int rows,
    int h,
    int intermediate,
    float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const __half* x_row = x + row * h;
    __shared__ float smem[BLOCK_SIZE_512 / WARP_SIZE];

    float local_sq = 0.0f;
    int vec_cols = h / 2;
    int remainder = h % 2;
    const __half2* x_row2 = reinterpret_cast<const __half2*>(x_row);

    for (int i = threadIdx.x; i < vec_cols; i += BLOCK_SIZE_512) {
        __half2 v = x_row2[i];
        float v0 = __half2float(v.x), v1 = __half2float(v.y);
        local_sq += v0 * v0 + v1 * v1;
    }
    if (remainder && threadIdx.x == 0) {
        float v = __half2float(x_row[h - 1]);
        local_sq += v * v;
    }

    float total_sq;
    {
        int lane = threadIdx.x % WARP_SIZE;
        int warp_id = threadIdx.x / WARP_SIZE;
        local_sq = warp_reduce_sum(local_sq);
        if (lane == 0) smem[warp_id] = local_sq;
        __syncthreads();
        constexpr int NUM_WARPS = BLOCK_SIZE_512 / WARP_SIZE;
        if (warp_id == 0) {
            local_sq = (lane < NUM_WARPS) ? smem[lane] : 0.0f;
            local_sq = warp_reduce_sum(local_sq);
        }
        total_sq = local_sq;
    }

    __shared__ float rms_shared;
    if (threadIdx.x == 0) {
        rms_shared = sqrtf(total_sq / (float)h + eps);
    }
    __syncthreads();
    float inv_rms = 1.0f / rms_shared;

    const __half* gate_row = raw_output + row * 2 * intermediate;
    const __half* up_row   = gate_row + intermediate;
    __half* out_row        = output + row * intermediate;
    const __half* b_gate   = b_new;
    const __half* b_up     = b_new + intermediate;

    int out_vec = intermediate / 2;
    int out_rem = intermediate % 2;

    for (int i = threadIdx.x; i < out_vec; i += BLOCK_SIZE_512) {
        __half2 g = reinterpret_cast<const __half2*>(gate_row)[i];
        __half2 u = reinterpret_cast<const __half2*>(up_row)[i];
        __half2 bg = reinterpret_cast<const __half2*>(b_gate)[i];
        __half2 bu = reinterpret_cast<const __half2*>(b_up)[i];
        float gv0 = __half2float(g.x) * inv_rms + __half2float(bg.x);
        float uv0 = __half2float(u.x) * inv_rms + __half2float(bu.x);
        float r0 = (gv0 / (1.0f + expf(-gv0))) * uv0;
        float gv1 = __half2float(g.y) * inv_rms + __half2float(bg.y);
        float uv1 = __half2float(u.y) * inv_rms + __half2float(bu.y);
        float r1 = (gv1 / (1.0f + expf(-gv1))) * uv1;
        reinterpret_cast<__half2*>(out_row)[i] = __halves2half2(__float2half(r0), __float2half(r1));
    }
    int out_base = out_vec * 2;
    for (int i = threadIdx.x; i < out_rem; i += BLOCK_SIZE_512) {
        int c = out_base + i;
        float gv = __half2float(gate_row[c]) * inv_rms + __half2float(b_gate[c]);
        float uv = __half2float(up_row[c]) * inv_rms + __half2float(b_up[c]);
        out_row[c] = __float2half((gv / (1.0f + expf(-gv))) * uv);
    }
}

// V3 RMSNorm+SwiGLU BF16: 512 threads
__global__ void rmsnorm_swiglu_512_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ raw_output,
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ b_new,
    int rows,
    int h,
    int intermediate,
    float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const __nv_bfloat16* x_row = x + row * h;
    __shared__ float smem[BLOCK_SIZE_512 / WARP_SIZE];

    float local_sq = 0.0f;
    int vec_cols = h / 2;
    int remainder = h % 2;
    const __nv_bfloat162* x_row2 = reinterpret_cast<const __nv_bfloat162*>(x_row);

    for (int i = threadIdx.x; i < vec_cols; i += BLOCK_SIZE_512) {
        __nv_bfloat162 v = x_row2[i];
        float v0 = __bfloat162float(v.x), v1 = __bfloat162float(v.y);
        local_sq += v0 * v0 + v1 * v1;
    }
    if (remainder && threadIdx.x == 0) {
        float v = __bfloat162float(x_row[h - 1]);
        local_sq += v * v;
    }

    float total_sq;
    {
        int lane = threadIdx.x % WARP_SIZE;
        int warp_id = threadIdx.x / WARP_SIZE;
        local_sq = warp_reduce_sum(local_sq);
        if (lane == 0) smem[warp_id] = local_sq;
        __syncthreads();
        constexpr int NUM_WARPS = BLOCK_SIZE_512 / WARP_SIZE;
        if (warp_id == 0) {
            local_sq = (lane < NUM_WARPS) ? smem[lane] : 0.0f;
            local_sq = warp_reduce_sum(local_sq);
        }
        total_sq = local_sq;
    }

    __shared__ float rms_shared;
    if (threadIdx.x == 0) {
        rms_shared = sqrtf(total_sq / (float)h + eps);
    }
    __syncthreads();
    float inv_rms = 1.0f / rms_shared;

    const __nv_bfloat16* gate_row = raw_output + row * 2 * intermediate;
    const __nv_bfloat16* up_row   = gate_row + intermediate;
    __nv_bfloat16* out_row        = output + row * intermediate;
    const __nv_bfloat16* b_gate   = b_new;
    const __nv_bfloat16* b_up     = b_new + intermediate;

    int out_vec = intermediate / 2;
    int out_rem = intermediate % 2;

    for (int i = threadIdx.x; i < out_vec; i += BLOCK_SIZE_512) {
        __nv_bfloat162 g = reinterpret_cast<const __nv_bfloat162*>(gate_row)[i];
        __nv_bfloat162 u = reinterpret_cast<const __nv_bfloat162*>(up_row)[i];
        __nv_bfloat162 bg = reinterpret_cast<const __nv_bfloat162*>(b_gate)[i];
        __nv_bfloat162 bu = reinterpret_cast<const __nv_bfloat162*>(b_up)[i];
        float gv0 = __bfloat162float(g.x) * inv_rms + __bfloat162float(bg.x);
        float uv0 = __bfloat162float(u.x) * inv_rms + __bfloat162float(bu.x);
        float r0 = (gv0 / (1.0f + expf(-gv0))) * uv0;
        float gv1 = __bfloat162float(g.y) * inv_rms + __bfloat162float(bg.y);
        float uv1 = __bfloat162float(u.y) * inv_rms + __bfloat162float(bu.y);
        float r1 = (gv1 / (1.0f + expf(-gv1))) * uv1;
        reinterpret_cast<__nv_bfloat162*>(out_row)[i] = __halves2bfloat162(__float2bfloat16(r0), __float2bfloat16(r1));
    }
    int out_base = out_vec * 2;
    for (int i = threadIdx.x; i < out_rem; i += BLOCK_SIZE_512) {
        int c = out_base + i;
        float gv = __bfloat162float(gate_row[c]) * inv_rms + __bfloat162float(b_gate[c]);
        float uv = __bfloat162float(up_row[c]) * inv_rms + __bfloat162float(b_up[c]);
        out_row[c] = __float2bfloat16((gv / (1.0f + expf(-gv))) * uv);
    }
}

// ============================================================================
// Host functions
// ============================================================================

// Host function: dispatches fp32 or fp16 kernel
torch::Tensor compute_denominator_cuda(
    torch::Tensor x,
    c10::optional<int64_t> stream_ptr
) {
    TORCH_CHECK(x.dim() == 2, "Input must be 2D");
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA");

    int rows = x.size(0);
    int cols = x.size(1);

    auto output = torch::empty({rows}, torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));

    cudaStream_t stream;
    if (stream_ptr.has_value()) {
        stream = reinterpret_cast<cudaStream_t>(stream_ptr.value());
    } else {
        stream = at::cuda::getCurrentCUDAStream().stream();
    }

    dim3 grid(rows);
    dim3 block(BLOCK_SIZE);

    if (x.scalar_type() == torch::kFloat32) {
        denominator_fp32_kernel<<<grid, block, 0, stream>>>(
            x.data_ptr<float>(),
            output.data_ptr<float>(),
            rows, cols
        );
    } else if (x.scalar_type() == torch::kFloat16) {
        denominator_fp16_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            output.data_ptr<float>(),
            rows, cols
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype: only float32 and float16 are supported");
    }

    return output;
}

// V1: Fused denominator + normalize (in-place on raw_output)
void denominator_normalize_cuda(
    torch::Tensor x,
    torch::Tensor raw_output,
    torch::Tensor b_new,
    int h,
    float eps
) {
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(raw_output.dim() == 2, "raw_output must be 2D");
    TORCH_CHECK(x.is_cuda() && raw_output.is_cuda() && b_new.is_cuda(), "All tensors must be on CUDA");

    int rows = x.size(0);
    int out_dim = raw_output.size(1);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    dim3 grid(rows);
    dim3 block(BLOCK_SIZE);

    if (x.scalar_type() == torch::kFloat32) {
        denominator_normalize_fp32_kernel<<<grid, block, 0, stream>>>(
            x.data_ptr<float>(),
            raw_output.data_ptr<float>(),
            b_new.data_ptr<float>(),
            rows, h, out_dim, eps
        );
    } else if (x.scalar_type() == torch::kFloat16) {
        denominator_normalize_fp16_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(raw_output.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(b_new.data_ptr<at::Half>()),
            rows, h, out_dim, eps
        );
    } else if (x.scalar_type() == torch::kBFloat16) {
        denominator_normalize_bf16_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(raw_output.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(b_new.data_ptr<at::BFloat16>()),
            rows, h, out_dim, eps
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype for V1: only fp32, fp16, bf16 supported");
    }
}

// V2: Welford's single-pass denominator
torch::Tensor compute_denominator_welford_cuda(
    torch::Tensor x,
    c10::optional<int64_t> stream_ptr
) {
    TORCH_CHECK(x.dim() == 2, "Input must be 2D");
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Only fp32 supported for Welford");

    int rows = x.size(0);
    int cols = x.size(1);

    auto output = torch::empty({rows}, torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));

    cudaStream_t stream;
    if (stream_ptr.has_value()) {
        stream = reinterpret_cast<cudaStream_t>(stream_ptr.value());
    } else {
        stream = at::cuda::getCurrentCUDAStream().stream();
    }

    dim3 grid(rows);
    dim3 block(BLOCK_SIZE);

    denominator_welford_fp32_kernel<<<grid, block, 0, stream>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        rows, cols
    );

    return output;
}

// V3: Combined Welford + fused normalize + 512 threads
void denominator_normalize_welford_512_cuda(
    torch::Tensor x,
    torch::Tensor raw_output,
    torch::Tensor b_new,
    int h,
    float eps
) {
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(raw_output.dim() == 2, "raw_output must be 2D");
    TORCH_CHECK(x.is_cuda() && raw_output.is_cuda() && b_new.is_cuda(), "All tensors must be on CUDA");

    int rows = x.size(0);
    int out_dim = raw_output.size(1);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    dim3 grid(rows);
    dim3 block(BLOCK_SIZE_512);

    if (x.scalar_type() == torch::kFloat32) {
        denominator_normalize_welford_512_fp32_kernel<<<grid, block, 0, stream>>>(
            x.data_ptr<float>(),
            raw_output.data_ptr<float>(),
            b_new.data_ptr<float>(),
            rows, h, out_dim, eps
        );
    } else if (x.scalar_type() == torch::kFloat16) {
        denominator_normalize_welford_512_fp16_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(raw_output.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(b_new.data_ptr<at::Half>()),
            rows, h, out_dim, eps
        );
    } else if (x.scalar_type() == torch::kBFloat16) {
        denominator_normalize_welford_512_bf16_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(raw_output.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(b_new.data_ptr<at::BFloat16>()),
            rows, h, out_dim, eps
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype for V3: only fp32, fp16, bf16 supported");
    }
}

// RMSNorm V1: fused normalize (256 threads)
void rmsnorm_normalize_cuda(
    torch::Tensor x,
    torch::Tensor raw_output,
    torch::Tensor b_new,
    int h,
    float eps
) {
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(raw_output.dim() == 2, "raw_output must be 2D");
    TORCH_CHECK(x.is_cuda() && raw_output.is_cuda() && b_new.is_cuda(), "All tensors must be on CUDA");

    int rows = x.size(0);
    int out_dim = raw_output.size(1);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    dim3 grid(rows);
    dim3 block(BLOCK_SIZE);

    if (x.scalar_type() == torch::kFloat32) {
        rmsnorm_normalize_fp32_kernel<<<grid, block, 0, stream>>>(
            x.data_ptr<float>(),
            raw_output.data_ptr<float>(),
            b_new.data_ptr<float>(),
            rows, h, out_dim, eps
        );
    } else if (x.scalar_type() == torch::kFloat16) {
        rmsnorm_normalize_fp16_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(raw_output.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(b_new.data_ptr<at::Half>()),
            rows, h, out_dim, eps
        );
    } else if (x.scalar_type() == torch::kBFloat16) {
        rmsnorm_normalize_bf16_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(raw_output.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(b_new.data_ptr<at::BFloat16>()),
            rows, h, out_dim, eps
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype for RMSNorm V1: only fp32, fp16, bf16 supported");
    }
}

// RMSNorm V3: fused normalize (512 threads)
void rmsnorm_normalize_512_cuda(
    torch::Tensor x,
    torch::Tensor raw_output,
    torch::Tensor b_new,
    int h,
    float eps
) {
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(raw_output.dim() == 2, "raw_output must be 2D");
    TORCH_CHECK(x.is_cuda() && raw_output.is_cuda() && b_new.is_cuda(), "All tensors must be on CUDA");

    int rows = x.size(0);
    int out_dim = raw_output.size(1);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    dim3 grid(rows);
    dim3 block(BLOCK_SIZE_512);

    if (x.scalar_type() == torch::kFloat32) {
        rmsnorm_normalize_512_fp32_kernel<<<grid, block, 0, stream>>>(
            x.data_ptr<float>(),
            raw_output.data_ptr<float>(),
            b_new.data_ptr<float>(),
            rows, h, out_dim, eps
        );
    } else if (x.scalar_type() == torch::kFloat16) {
        rmsnorm_normalize_512_fp16_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(raw_output.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(b_new.data_ptr<at::Half>()),
            rows, h, out_dim, eps
        );
    } else if (x.scalar_type() == torch::kBFloat16) {
        rmsnorm_normalize_512_bf16_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(raw_output.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(b_new.data_ptr<at::BFloat16>()),
            rows, h, out_dim, eps
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype for RMSNorm V3: only fp32, fp16, bf16 supported");
    }
}

// RMSNorm+SwiGLU V1: fused normalize + SiLU + multiply (256 threads)
void rmsnorm_swiglu_cuda(
    torch::Tensor x,
    torch::Tensor raw_output,
    torch::Tensor output,
    torch::Tensor b_new,
    int h,
    int intermediate,
    float eps
) {
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(raw_output.dim() == 2, "raw_output must be 2D");
    TORCH_CHECK(output.dim() == 2, "output must be 2D");
    TORCH_CHECK(x.is_cuda() && raw_output.is_cuda() && output.is_cuda() && b_new.is_cuda(),
                "All tensors must be on CUDA");

    int rows = x.size(0);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    dim3 grid(rows);
    dim3 block(BLOCK_SIZE);

    if (x.scalar_type() == torch::kFloat32) {
        rmsnorm_swiglu_fp32_kernel<<<grid, block, 0, stream>>>(
            x.data_ptr<float>(),
            raw_output.data_ptr<float>(),
            output.data_ptr<float>(),
            b_new.data_ptr<float>(),
            rows, h, intermediate, eps
        );
    } else if (x.scalar_type() == torch::kFloat16) {
        rmsnorm_swiglu_fp16_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(raw_output.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(b_new.data_ptr<at::Half>()),
            rows, h, intermediate, eps
        );
    } else if (x.scalar_type() == torch::kBFloat16) {
        rmsnorm_swiglu_bf16_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(raw_output.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(b_new.data_ptr<at::BFloat16>()),
            rows, h, intermediate, eps
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype for RMSNorm+SwiGLU V1: only fp32, fp16, bf16 supported");
    }
}

// RMSNorm+SwiGLU V3: fused normalize + SiLU + multiply (512 threads)
void rmsnorm_swiglu_512_cuda(
    torch::Tensor x,
    torch::Tensor raw_output,
    torch::Tensor output,
    torch::Tensor b_new,
    int h,
    int intermediate,
    float eps
) {
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(raw_output.dim() == 2, "raw_output must be 2D");
    TORCH_CHECK(output.dim() == 2, "output must be 2D");
    TORCH_CHECK(x.is_cuda() && raw_output.is_cuda() && output.is_cuda() && b_new.is_cuda(),
                "All tensors must be on CUDA");

    int rows = x.size(0);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    dim3 grid(rows);
    dim3 block(BLOCK_SIZE_512);

    if (x.scalar_type() == torch::kFloat32) {
        rmsnorm_swiglu_512_fp32_kernel<<<grid, block, 0, stream>>>(
            x.data_ptr<float>(),
            raw_output.data_ptr<float>(),
            output.data_ptr<float>(),
            b_new.data_ptr<float>(),
            rows, h, intermediate, eps
        );
    } else if (x.scalar_type() == torch::kFloat16) {
        rmsnorm_swiglu_512_fp16_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(raw_output.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(b_new.data_ptr<at::Half>()),
            rows, h, intermediate, eps
        );
    } else if (x.scalar_type() == torch::kBFloat16) {
        rmsnorm_swiglu_512_bf16_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(raw_output.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(b_new.data_ptr<at::BFloat16>()),
            rows, h, intermediate, eps
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype for RMSNorm+SwiGLU V3: only fp32, fp16, bf16 supported");
    }
}
