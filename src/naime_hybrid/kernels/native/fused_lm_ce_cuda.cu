#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cfloat>
#include <vector>

static inline int next_power_of_two_host(int value) {
  int out = 1;
  while (out < value) {
    out <<= 1;
  }
  return out;
}

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

template <typename scalar_t>
__global__ void cross_entropy_forward_kernel(
    const scalar_t* __restrict__ logits,
    const int64_t* __restrict__ labels,
    float* __restrict__ losses,
    int64_t* __restrict__ valid_counts,
    int64_t rows,
    int64_t vocab,
    int64_t ignore_index) {
  extern __shared__ float shared[];
  float* reduce = shared;
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

  const int64_t base = row * vocab;
  float local_max = -FLT_MAX;
  for (int64_t v = tid; v < vocab; v += blockDim.x) {
    local_max = fmaxf(local_max, static_cast<float>(logits[base + v]));
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
  for (int64_t v = tid; v < vocab; v += blockDim.x) {
    local_sum += expf(static_cast<float>(logits[base + v]) - row_max);
  }
  reduce[tid] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] += reduce[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    const float target = static_cast<float>(logits[base + label]);
    losses[row] = logf(reduce[0]) + row_max - target;
    valid_counts[row] = 1;
  }
}

std::vector<at::Tensor> cross_entropy_forward_cuda(
    at::Tensor logits,
    at::Tensor labels,
    int64_t ignore_index) {
  const c10::cuda::CUDAGuard device_guard(logits.device());
  const auto rows = logits.size(0);
  const auto vocab = logits.size(1);

  auto losses = at::empty({rows}, logits.options().dtype(at::kFloat));
  auto valid_counts = at::empty({rows}, labels.options());

  const int threads = 256;
  const dim3 blocks(rows);
  const size_t shared_bytes = static_cast<size_t>(threads) * sizeof(float);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      logits.scalar_type(),
      "cross_entropy_forward_cuda",
      [&] {
        cross_entropy_forward_kernel<scalar_t><<<blocks, threads, shared_bytes, stream>>>(
            logits.data_ptr<scalar_t>(),
            labels.data_ptr<int64_t>(),
            losses.data_ptr<float>(),
            valid_counts.data_ptr<int64_t>(),
            rows,
            vocab,
            ignore_index);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto valid = valid_counts.sum();
  auto loss = losses.sum() / valid.clamp_min(1).to(losses.dtype());
  return {loss, valid};
}

template <typename scalar_t>
__global__ void cross_entropy_backward_kernel(
    const scalar_t* __restrict__ logits,
    const int64_t* __restrict__ labels,
    const int64_t* __restrict__ valid_count,
    const float* __restrict__ grad_output,
    scalar_t* __restrict__ grad_logits,
    int64_t rows,
    int64_t vocab,
    int64_t ignore_index) {
  extern __shared__ float shared[];
  float* reduce = shared;
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  const int64_t label = labels[row];
  const int64_t base = row * vocab;

  if (label == ignore_index) {
    for (int64_t v = tid; v < vocab; v += blockDim.x) {
      grad_logits[base + v] = static_cast<scalar_t>(0.0f);
    }
    return;
  }

  float local_max = -FLT_MAX;
  for (int64_t v = tid; v < vocab; v += blockDim.x) {
    local_max = fmaxf(local_max, static_cast<float>(logits[base + v]));
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
  for (int64_t v = tid; v < vocab; v += blockDim.x) {
    local_sum += expf(static_cast<float>(logits[base + v]) - row_max);
  }
  reduce[tid] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] += reduce[tid + stride];
    }
    __syncthreads();
  }
  const float denom = reduce[0];
  const int64_t valid = valid_count[0] > 1 ? valid_count[0] : 1;
  const float scale = grad_output[0] / static_cast<float>(valid);

  for (int64_t v = tid; v < vocab; v += blockDim.x) {
    float grad = expf(static_cast<float>(logits[base + v]) - row_max) / denom;
    if (v == label) {
      grad -= 1.0f;
    }
    grad_logits[base + v] = static_cast<scalar_t>(grad * scale);
  }
}

at::Tensor cross_entropy_backward_cuda(
    at::Tensor grad_output,
    at::Tensor logits,
    at::Tensor labels,
    at::Tensor valid_count,
    int64_t ignore_index) {
  const c10::cuda::CUDAGuard device_guard(logits.device());
  const auto rows = logits.size(0);
  const auto vocab = logits.size(1);
  auto grad_logits = at::empty_like(logits);

  const int threads = 256;
  const dim3 blocks(rows);
  const size_t shared_bytes = static_cast<size_t>(threads) * sizeof(float);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      logits.scalar_type(),
      "cross_entropy_backward_cuda",
      [&] {
        cross_entropy_backward_kernel<scalar_t><<<blocks, threads, shared_bytes, stream>>>(
            logits.data_ptr<scalar_t>(),
            labels.data_ptr<int64_t>(),
            valid_count.data_ptr<int64_t>(),
            grad_output.data_ptr<float>(),
            grad_logits.data_ptr<scalar_t>(),
            rows,
            vocab,
            ignore_index);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_logits;
}

template <typename scalar_t>
__global__ void fused_rms_norm_forward_kernel(
    const scalar_t* __restrict__ input,
    const float* __restrict__ weight,
    scalar_t* __restrict__ output,
    float* __restrict__ inv_rms,
    int64_t rows,
    int64_t dim,
    float eps) {
  extern __shared__ float shared[];
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  const int64_t base = row * dim;

  float sumsq = 0.0f;
  for (int64_t d = tid; d < dim; d += blockDim.x) {
    const float x = static_cast<float>(input[base + d]);
    sumsq += x * x;
  }
  shared[tid] = sumsq;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }

  const float row_inv = rsqrtf(shared[0] / static_cast<float>(dim) + eps);
  if (tid == 0) {
    inv_rms[row] = row_inv;
  }

  for (int64_t d = tid; d < dim; d += blockDim.x) {
    const float x = static_cast<float>(input[base + d]);
    const float y = x * row_inv * weight[d];
    output[base + d] = static_cast<scalar_t>(y);
  }
}

template <typename scalar_t>
__global__ void fused_rms_norm_backward_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ inv_rms,
    scalar_t* __restrict__ grad_input,
    float* __restrict__ grad_weight,
    int64_t rows,
    int64_t dim) {
  extern __shared__ float shared[];
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  const int64_t base = row * dim;
  const float row_inv = inv_rms[row];

  float dot = 0.0f;
  for (int64_t d = tid; d < dim; d += blockDim.x) {
    const float go = static_cast<float>(grad_output[base + d]);
    const float x = static_cast<float>(input[base + d]);
    dot += go * weight[d] * x;
  }
  shared[tid] = dot;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  const float row_dot = shared[0];
  const float correction_scale = row_inv * row_inv * row_dot / static_cast<float>(dim);

  for (int64_t d = tid; d < dim; d += blockDim.x) {
    const float go = static_cast<float>(grad_output[base + d]);
    const float x = static_cast<float>(input[base + d]);
    const float u = go * weight[d];
    const float dx = row_inv * (u - x * correction_scale);
    grad_input[base + d] = static_cast<scalar_t>(dx);
    atomicAdd(&grad_weight[d], go * x * row_inv);
  }
}

std::vector<at::Tensor> fused_rms_norm_forward_cuda(
    at::Tensor input,
    at::Tensor weight,
    double eps) {
  const c10::cuda::CUDAGuard device_guard(input.device());
  const auto dim = input.size(-1);
  const auto rows = input.numel() / dim;
  auto output = at::empty_like(input);
  auto inv_rms = at::empty({rows}, input.options().dtype(at::kFloat));

  const int threads = std::min(1024, std::max(32, next_power_of_two_host(static_cast<int>(dim))));
  const dim3 blocks(rows);
  const size_t shared_bytes = static_cast<size_t>(threads) * sizeof(float);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "fused_rms_norm_forward_cuda",
      [&] {
        fused_rms_norm_forward_kernel<scalar_t><<<blocks, threads, shared_bytes, stream>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<float>(),
            output.data_ptr<scalar_t>(),
            inv_rms.data_ptr<float>(),
            rows,
            dim,
            static_cast<float>(eps));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, inv_rms};
}

std::vector<at::Tensor> fused_rms_norm_backward_cuda(
    at::Tensor grad_output,
    at::Tensor input,
    at::Tensor weight,
    at::Tensor inv_rms) {
  const c10::cuda::CUDAGuard device_guard(input.device());
  const auto dim = input.size(-1);
  const auto rows = input.numel() / dim;
  auto grad_input = at::empty_like(input);
  auto grad_weight = at::zeros_like(weight, weight.options().dtype(at::kFloat));

  const int threads = std::min(1024, std::max(32, next_power_of_two_host(static_cast<int>(dim))));
  const dim3 blocks(rows);
  const size_t shared_bytes = static_cast<size_t>(threads) * sizeof(float);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "fused_rms_norm_backward_cuda",
      [&] {
        fused_rms_norm_backward_kernel<scalar_t><<<blocks, threads, shared_bytes, stream>>>(
            grad_output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            weight.data_ptr<float>(),
            inv_rms.data_ptr<float>(),
            grad_input.data_ptr<scalar_t>(),
            grad_weight.data_ptr<float>(),
            rows,
            dim);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_input, grad_weight};
}

template <typename scalar_t>
__global__ void fused_state_softmax_matmul_forward_kernel(
    const scalar_t* __restrict__ scores,
    const scalar_t* __restrict__ values,
    const bool* __restrict__ mask,
    scalar_t* __restrict__ context,
    scalar_t* __restrict__ weights,
    int64_t batch,
    int64_t tokens,
    int64_t slots,
    int64_t dim,
    int64_t mask_batch,
    bool has_mask,
    bool zero_invalid) {
  extern __shared__ float shared[];
  float* reduce = shared;
  float* row_weights = shared + blockDim.x;

  const int64_t row = blockIdx.x;
  const int64_t b = row / tokens;
  const int64_t t = row - b * tokens;
  const int tid = threadIdx.x;
  const int64_t score_base = (b * tokens + t) * slots;
  const int64_t mask_b = mask_batch == 1 ? 0 : b;
  const int64_t mask_base = (mask_b * tokens + t) * slots;

  int valid_count = 0;
  float local_max = -FLT_MAX;
  for (int64_t s = tid; s < slots; s += blockDim.x) {
    const bool masked = has_mask && mask[mask_base + s];
    if (!masked) {
      const float score = static_cast<float>(scores[score_base + s]);
      local_max = fmaxf(local_max, score);
      valid_count += 1;
    }
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

  reduce[tid] = static_cast<float>(valid_count);
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] += reduce[tid + stride];
    }
    __syncthreads();
  }
  const bool row_valid = reduce[0] > 0.0f;

  float local_sum = 0.0f;
  for (int64_t s = tid; s < slots; s += blockDim.x) {
    const bool masked = has_mask && mask[mask_base + s];
    float w = 0.0f;
    if (row_valid && !masked) {
      w = expf(static_cast<float>(scores[score_base + s]) - row_max);
      local_sum += w;
    }
    row_weights[s] = w;
  }
  reduce[tid] = local_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] += reduce[tid + stride];
    }
    __syncthreads();
  }
  const float denom = reduce[0];

  for (int64_t s = tid; s < slots; s += blockDim.x) {
    float w = (row_valid && denom > 0.0f) ? row_weights[s] / denom : 0.0f;
    if (!zero_invalid && !row_valid) {
      w = 0.0f;
    }
    row_weights[s] = w;
    weights[score_base + s] = static_cast<scalar_t>(w);
  }
  __syncthreads();

  const int64_t context_base = (b * tokens + t) * dim;
  const int64_t value_base = b * slots * dim;
  for (int64_t d = tid; d < dim; d += blockDim.x) {
    float acc = 0.0f;
    for (int64_t s = 0; s < slots; ++s) {
      acc += row_weights[s] * static_cast<float>(values[value_base + s * dim + d]);
    }
    context[context_base + d] = static_cast<scalar_t>(acc);
  }
}

std::vector<at::Tensor> fused_state_softmax_matmul_forward_cuda(
    at::Tensor scores,
    at::Tensor values,
    at::Tensor mask,
    bool zero_invalid) {
  const c10::cuda::CUDAGuard device_guard(scores.device());
  const auto batch = scores.size(0);
  const auto tokens = scores.size(1);
  const auto slots = scores.size(2);
  const auto dim = values.size(2);
  const bool has_mask = mask.numel() > 0;
  const auto mask_batch = has_mask ? mask.size(0) : 0;

  auto context = at::empty({batch, tokens, dim}, scores.options());
  auto weights = at::empty_like(scores);

  const int threads = std::min(1024, std::max(32, next_power_of_two_host(static_cast<int>(std::max(slots, dim)))));
  const dim3 blocks(batch * tokens);
  const size_t shared_bytes = static_cast<size_t>(threads + slots) * sizeof(float);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      scores.scalar_type(),
      "fused_state_softmax_matmul_forward_cuda",
      [&] {
        fused_state_softmax_matmul_forward_kernel<scalar_t><<<blocks, threads, shared_bytes, stream>>>(
            scores.data_ptr<scalar_t>(),
            values.data_ptr<scalar_t>(),
            has_mask ? mask.data_ptr<bool>() : nullptr,
            context.data_ptr<scalar_t>(),
            weights.data_ptr<scalar_t>(),
            batch,
            tokens,
            slots,
            dim,
            mask_batch,
            has_mask,
            zero_invalid);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {context, weights};
}

template <typename scalar_t>
__global__ void fused_state_softmax_matmul_grad_scores_kernel(
    const scalar_t* __restrict__ grad_context,
    const scalar_t* __restrict__ grad_weights,
    const scalar_t* __restrict__ weights,
    const scalar_t* __restrict__ values,
    const bool* __restrict__ mask,
    scalar_t* __restrict__ grad_scores,
    int64_t batch,
    int64_t tokens,
    int64_t slots,
    int64_t dim,
    bool has_grad_weights,
    bool has_mask) {
  extern __shared__ float shared[];
  float* grad_w = shared;
  float* reduce = shared + slots;

  const int64_t row = blockIdx.x;
  const int64_t b = row / tokens;
  const int64_t t = row - b * tokens;
  const int tid = threadIdx.x;
  const int64_t weight_base = (b * tokens + t) * slots;
  const int64_t grad_context_base = (b * tokens + t) * dim;
  const int64_t value_base = b * slots * dim;

  for (int64_t s = tid; s < slots; s += blockDim.x) {
    float dot = 0.0f;
    const int64_t slot_value_base = value_base + s * dim;
    for (int64_t d = 0; d < dim; ++d) {
      dot += static_cast<float>(grad_context[grad_context_base + d]) *
             static_cast<float>(values[slot_value_base + d]);
    }
    if (has_grad_weights) {
      dot += static_cast<float>(grad_weights[weight_base + s]);
    }
    grad_w[s] = dot;
  }
  __syncthreads();

  float local_sum = 0.0f;
  for (int64_t s = tid; s < slots; s += blockDim.x) {
    local_sum += grad_w[s] * static_cast<float>(weights[weight_base + s]);
  }
  reduce[tid] = local_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] += reduce[tid + stride];
    }
    __syncthreads();
  }
  const float dot = reduce[0];

  for (int64_t s = tid; s < slots; s += blockDim.x) {
    const bool masked = has_mask && mask[weight_base + s];
    const float w = static_cast<float>(weights[weight_base + s]);
    const float ds = masked ? 0.0f : w * (grad_w[s] - dot);
    grad_scores[weight_base + s] = static_cast<scalar_t>(ds);
  }
}

template <typename scalar_t>
__global__ void fused_state_softmax_matmul_grad_values_kernel(
    const scalar_t* __restrict__ grad_context,
    const scalar_t* __restrict__ weights,
    scalar_t* __restrict__ grad_values,
    int64_t batch,
    int64_t tokens,
    int64_t slots,
    int64_t dim) {
  const int64_t row = blockIdx.x;
  const int64_t b = row / slots;
  const int64_t s = row - b * slots;
  const int tid = threadIdx.x;
  const int64_t grad_value_base = (b * slots + s) * dim;
  const int64_t weight_base = b * tokens * slots + s;
  const int64_t grad_context_base = b * tokens * dim;

  for (int64_t d = tid; d < dim; d += blockDim.x) {
    float acc = 0.0f;
    for (int64_t t = 0; t < tokens; ++t) {
      acc += static_cast<float>(weights[weight_base + t * slots]) *
             static_cast<float>(grad_context[grad_context_base + t * dim + d]);
    }
    grad_values[grad_value_base + d] = static_cast<scalar_t>(acc);
  }
}

std::vector<at::Tensor> fused_state_softmax_matmul_backward_cuda(
    at::Tensor grad_context,
    at::Tensor grad_weights,
    at::Tensor weights,
    at::Tensor values,
    at::Tensor mask) {
  const c10::cuda::CUDAGuard device_guard(weights.device());
  const auto batch = weights.size(0);
  const auto tokens = weights.size(1);
  const auto slots = weights.size(2);
  const auto dim = values.size(2);
  const bool has_grad_weights = grad_weights.numel() > 0;
  const bool has_mask = mask.numel() > 0;

  auto grad_scores = at::empty_like(weights);
  auto grad_values = at::empty_like(values);

  const int threads_scores = std::min(1024, std::max(32, next_power_of_two_host(static_cast<int>(slots))));
  const int threads_values = std::min(1024, std::max(32, next_power_of_two_host(static_cast<int>(dim))));
  const dim3 blocks_scores(batch * tokens);
  const dim3 blocks_values(batch * slots);
  const size_t shared_scores = static_cast<size_t>(slots + threads_scores) * sizeof(float);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      weights.scalar_type(),
      "fused_state_softmax_matmul_backward_cuda",
      [&] {
        fused_state_softmax_matmul_grad_scores_kernel<scalar_t><<<
            blocks_scores,
            threads_scores,
            shared_scores,
            stream>>>(
            grad_context.data_ptr<scalar_t>(),
            has_grad_weights ? grad_weights.data_ptr<scalar_t>() : nullptr,
            weights.data_ptr<scalar_t>(),
            values.data_ptr<scalar_t>(),
            has_mask ? mask.data_ptr<bool>() : nullptr,
            grad_scores.data_ptr<scalar_t>(),
            batch,
            tokens,
            slots,
            dim,
            has_grad_weights,
            has_mask);
        fused_state_softmax_matmul_grad_values_kernel<scalar_t><<<blocks_values, threads_values, 0, stream>>>(
            grad_context.data_ptr<scalar_t>(),
            weights.data_ptr<scalar_t>(),
            grad_values.data_ptr<scalar_t>(),
            batch,
            tokens,
            slots,
            dim);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_scores, grad_values};
}
