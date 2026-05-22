#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cfloat>
#include <vector>

template <typename scalar_t>
__global__ void fused_lm_ce_forward_kernel(
    const scalar_t* __restrict__ hidden,
    const scalar_t* __restrict__ weight,
    const int64_t* __restrict__ labels,
    float* __restrict__ losses,
    int64_t* __restrict__ valid_counts,
    int64_t rows,
    int64_t dim,
    int64_t vocab,
    int64_t ignore_index) {
  extern __shared__ float shared[];
  float* reduce = shared;
  float* target_reduce = shared + blockDim.x;
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;

  const int64_t label = labels[row];
  if (label == ignore_index) {
    if (tid == 0) {
      losses[row] = 0.0f;
      valid_counts[row] = 0;
    }
    return;
  }

  float local_max = -FLT_MAX;
  for (int64_t v = tid; v < vocab; v += blockDim.x) {
    float dot = 0.0f;
    const int64_t weight_base = v * dim;
    const int64_t hidden_base = row * dim;
    for (int64_t d = 0; d < dim; ++d) {
      dot += static_cast<float>(hidden[hidden_base + d]) * static_cast<float>(weight[weight_base + d]);
    }
    local_max = fmaxf(local_max, dot);
  }
  reduce[tid] = local_max;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
    }
    __syncthreads();
  }
  const float row_max = reduce[0];

  float local_sum = 0.0f;
  float local_target = 0.0f;
  for (int64_t v = tid; v < vocab; v += blockDim.x) {
    float dot = 0.0f;
    const int64_t weight_base = v * dim;
    const int64_t hidden_base = row * dim;
    for (int64_t d = 0; d < dim; ++d) {
      dot += static_cast<float>(hidden[hidden_base + d]) * static_cast<float>(weight[weight_base + d]);
    }
    local_sum += expf(dot - row_max);
    if (v == label) {
      local_target = dot;
    }
  }
  reduce[tid] = local_sum;
  target_reduce[tid] = local_target;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] += reduce[tid + stride];
      target_reduce[tid] += target_reduce[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    losses[row] = logf(reduce[0]) + row_max - target_reduce[0];
    valid_counts[row] = 1;
  }
}

std::vector<at::Tensor> fused_lm_ce_forward_cuda(
    at::Tensor hidden,
    at::Tensor weight,
    at::Tensor labels,
    int64_t ignore_index) {
  const c10::cuda::CUDAGuard device_guard(hidden.device());
  const auto rows = hidden.size(0);
  const auto dim = hidden.size(1);
  const auto vocab = weight.size(0);

  auto losses = at::empty({rows}, hidden.options().dtype(at::kFloat));
  auto valid_counts = at::empty({rows}, labels.options());

  const int threads = 256;
  const dim3 blocks(rows);
  const size_t shared_bytes = static_cast<size_t>(threads) * 2 * sizeof(float);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      hidden.scalar_type(),
      "fused_lm_ce_forward_cuda",
      [&] {
        fused_lm_ce_forward_kernel<scalar_t><<<blocks, threads, shared_bytes, stream>>>(
            hidden.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            labels.data_ptr<int64_t>(),
            losses.data_ptr<float>(),
            valid_counts.data_ptr<int64_t>(),
            rows,
            dim,
            vocab,
            ignore_index);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto valid = valid_counts.sum();
  auto loss = losses.sum() / valid.clamp_min(1).to(losses.dtype());
  return {loss, valid};
}
