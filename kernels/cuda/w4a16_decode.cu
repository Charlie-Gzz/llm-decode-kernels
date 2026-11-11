// W4A16 decode GEMV in CUDA C++ (batch M = 1..16). Same weight format as kernels/quant.py:
//   qweight int32 [K/8, N] (8 nibbles along K), scales/zeros fp16 [K/group, N].
//
// Work split:
//   - A block owns 128 output columns. Each lane owns 4 adjacent columns, so one 16-byte load
//     (int4) brings 4 packed int32 = 32 weights, and a warp reads 512 contiguous bytes.
//   - The 4 warps of a block split the block's K range by packed row; partial sums meet in
//     shared memory (shared-memory atomics), then the block writes its 128 columns.
//   - gridDim.y splits K across blocks (split-K) so small N still fills the GPU. With split-K the
//     blocks atomically add fp32 partials into a zeroed workspace that the host casts to fp16.
//   - Activations for a K tile are staged in shared memory as fp32 and read as broadcasts.
// Dequantization is factored like the Triton batch-1 kernel: inside one packed row the 8 weights
// share a group, so sum_i x_i * (q_i - z) * s = s * (sum_i x_i * q_i - z * sum_i x_i). The inner
// loop is shift, mask, convert and one FMA per weight; the row sums of x are precomputed once per
// tile in shared memory. The L4 is power-limited (72 W), so fewer ALU ops means higher clocks.

#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

namespace {

constexpr int kWarps = 4;
constexpr int kThreads = 32 * kWarps;
constexpr int kColsPerThread = 4;
constexpr int kColsPerBlock = 32 * kColsPerThread;  // 128
constexpr int kTileK = 256;                         // K values staged in shared memory per step

template <int MT>
__global__ void __launch_bounds__(kThreads) w4a16_decode_kernel(
    const half* __restrict__ x, const int* __restrict__ qweight, const half* __restrict__ scales,
    const half* __restrict__ zeros, const half* __restrict__ bias, half* __restrict__ out_f16,
    float* __restrict__ out_f32, int M, int N, int K, int group_size, int k_per_split) {
  __shared__ float xs[MT][kTileK];
  __shared__ float xsum8[MT][kTileK / 8];  // sum of each run of 8 activations (one packed row)
  __shared__ float red[MT][kColsPerBlock];

  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int n0 = blockIdx.x * kColsPerBlock + lane * kColsPerThread;
  const bool cols_ok = n0 < N;  // host checks N % 4 == 0, so a lane's 4 columns are all in or out
  const int k_begin = blockIdx.y * k_per_split;
  const int k_end = min(k_begin + k_per_split, K);

  for (int i = threadIdx.x; i < MT * kColsPerBlock; i += kThreads) (&red[0][0])[i] = 0.f;

  float acc[MT][kColsPerThread];
#pragma unroll
  for (int m = 0; m < MT; ++m)
#pragma unroll
    for (int c = 0; c < kColsPerThread; ++c) acc[m][c] = 0.f;

  for (int kt = k_begin; kt < k_end; kt += kTileK) {
    const int klen = min(kTileK, k_end - kt);  // multiple of 8
    __syncthreads();
    for (int i = threadIdx.x; i < MT * kTileK; i += kThreads) {
      const int m = i / kTileK, k = i % kTileK;
      xs[m][k] = (m < M && k < klen) ? __half2float(x[(size_t)m * K + kt + k]) : 0.f;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < MT * (kTileK / 8); i += kThreads) {
      const int m = i / (kTileK / 8), r = i % (kTileK / 8);
      float sum = 0.f;
#pragma unroll
      for (int j = 0; j < 8; ++j) sum += xs[m][r * 8 + j];
      xsum8[m][r] = sum;
    }
    __syncthreads();
    if (!cols_ok) continue;

    for (int r = warp; r < klen / 8; r += kWarps) {
      const int krow = kt / 8 + r;
      const int4 q = *reinterpret_cast<const int4*>(qweight + (size_t)krow * N + n0);
      const int g = (kt + r * 8) / group_size;
      const uint2 s2 = *reinterpret_cast<const uint2*>(scales + (size_t)g * N + n0);
      const uint2 z2 = *reinterpret_cast<const uint2*>(zeros + (size_t)g * N + n0);
      const half* sh = reinterpret_cast<const half*>(&s2);
      const half* zh = reinterpret_cast<const half*>(&z2);
      const int qs[kColsPerThread] = {q.x, q.y, q.z, q.w};

#pragma unroll
      for (int c = 0; c < kColsPerThread; ++c) {
        const float s = __half2float(sh[c]);
        const float z = __half2float(zh[c]);
        float xq[MT];
#pragma unroll
        for (int m = 0; m < MT; ++m) xq[m] = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const float qv = static_cast<float>((qs[c] >> (4 * i)) & 0xF);
#pragma unroll
          for (int m = 0; m < MT; ++m) xq[m] += qv * xs[m][r * 8 + i];
        }
#pragma unroll
        for (int m = 0; m < MT; ++m) acc[m][c] += s * (xq[m] - z * xsum8[m][r]);
      }
    }
  }

  if (cols_ok) {
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
      for (int c = 0; c < kColsPerThread; ++c)
        atomicAdd(&red[m][lane * kColsPerThread + c], acc[m][c]);
  }
  __syncthreads();

  for (int i = threadIdx.x; i < MT * kColsPerBlock; i += kThreads) {
    const int m = i / kColsPerBlock, col = i % kColsPerBlock;
    const int n = blockIdx.x * kColsPerBlock + col;
    if (m >= M || n >= N) continue;
    float v = red[m][col];
    if (bias != nullptr && blockIdx.y == 0) v += __half2float(bias[n]);
    if (gridDim.y == 1) {
      out_f16[(size_t)m * N + n] = __float2half_rn(v);
    } else {
      atomicAdd(&out_f32[(size_t)m * N + n], v);
    }
  }
}

template <int MT>
void launch(dim3 grid, cudaStream_t stream, const torch::Tensor& x, const torch::Tensor& qw,
            const torch::Tensor& s, const torch::Tensor& z, const half* bias, half* out16,
            float* out32, int M, int N, int K, int group_size, int k_per_split) {
  w4a16_decode_kernel<MT><<<grid, kThreads, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()), qw.data_ptr<int>(),
      reinterpret_cast<const half*>(s.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(z.data_ptr<at::Half>()), bias, out16, out32, M, N, K,
      group_size, k_per_split);
}

}  // namespace

torch::Tensor w4a16_decode(torch::Tensor x, torch::Tensor qweight, torch::Tensor scales,
                           torch::Tensor zeros, c10::optional<torch::Tensor> bias,
                           int64_t group_size, int64_t split_k) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && x.dim() == 2 && x.is_contiguous());
  TORCH_CHECK(qweight.scalar_type() == at::kInt && qweight.is_contiguous());
  TORCH_CHECK(scales.scalar_type() == at::kHalf && scales.is_contiguous());
  TORCH_CHECK(zeros.scalar_type() == at::kHalf && zeros.is_contiguous());
  const int M = x.size(0), K = x.size(1), N = qweight.size(1);
  TORCH_CHECK(M >= 1 && M <= 16, "decode kernel supports M <= 16");
  TORCH_CHECK(qweight.size(0) * 8 == K && K % group_size == 0 && group_size % 8 == 0);
  TORCH_CHECK(N % 4 == 0, "N must be a multiple of 4 (16-byte loads)");

  const half* bias_ptr = nullptr;
  if (bias.has_value()) {
    TORCH_CHECK(bias->scalar_type() == at::kHalf && bias->is_contiguous() && bias->numel() == N);
    bias_ptr = reinterpret_cast<const half*>(bias->data_ptr<at::Half>());
  }

  // K range per block, a multiple of 8 so packed rows never straddle two blocks.
  int k_per_split = (K + split_k - 1) / split_k;
  k_per_split = (k_per_split + 7) / 8 * 8;
  const int splits = (K + k_per_split - 1) / k_per_split;
  const dim3 grid((N + kColsPerBlock - 1) / kColsPerBlock, splits);
  auto stream = at::cuda::getCurrentCUDAStream();

  torch::Tensor out16, out32;
  if (splits == 1) {
    out16 = torch::empty({M, N}, x.options());
  } else {
    out32 = torch::zeros({M, N}, x.options().dtype(at::kFloat));
  }
  half* o16 = splits == 1 ? reinterpret_cast<half*>(out16.data_ptr<at::Half>()) : nullptr;
  float* o32 = splits == 1 ? nullptr : out32.data_ptr<float>();

#define LAUNCH(MT) \
  launch<MT>(grid, stream, x, qweight, scales, zeros, bias_ptr, o16, o32, M, N, K, group_size, \
             k_per_split)
  if (M == 1) LAUNCH(1);
  else if (M == 2) LAUNCH(2);
  else if (M <= 4) LAUNCH(4);
  else if (M <= 8) LAUNCH(8);
  else LAUNCH(16);
#undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return splits == 1 ? out16 : out32.to(at::kHalf);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("w4a16_decode", &w4a16_decode, "W4A16 decode GEMV (M <= 16)");
}
