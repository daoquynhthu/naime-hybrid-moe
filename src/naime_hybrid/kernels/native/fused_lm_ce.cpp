#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> fused_lm_ce_forward_cuda(
    torch::Tensor hidden,
    torch::Tensor weight,
    torch::Tensor labels,
    int64_t ignore_index);

std::vector<torch::Tensor> fused_rms_norm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    double eps);

std::vector<torch::Tensor> fused_rms_norm_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor inv_rms);

std::vector<torch::Tensor> fused_state_softmax_matmul_forward_cuda(
    torch::Tensor scores,
    torch::Tensor values,
    torch::Tensor mask,
    bool zero_invalid);

std::vector<torch::Tensor> fused_state_softmax_matmul_backward_cuda(
    torch::Tensor grad_context,
    torch::Tensor grad_weights,
    torch::Tensor weights,
    torch::Tensor values,
    torch::Tensor mask);

std::vector<torch::Tensor> fused_lm_ce_forward(
    torch::Tensor hidden,
    torch::Tensor weight,
    torch::Tensor labels,
    int64_t ignore_index) {
  TORCH_CHECK(hidden.is_cuda(), "hidden must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  TORCH_CHECK(labels.is_cuda(), "labels must be a CUDA tensor");
  TORCH_CHECK(hidden.dim() == 2, "hidden must be [N, D]");
  TORCH_CHECK(weight.dim() == 2, "weight must be [V, D]");
  TORCH_CHECK(labels.dim() == 1, "labels must be [N]");
  TORCH_CHECK(hidden.size(0) == labels.size(0), "hidden rows must match labels");
  TORCH_CHECK(hidden.size(1) == weight.size(1), "hidden dim must match weight dim");
  TORCH_CHECK(labels.scalar_type() == torch::kInt64, "labels must be int64");
  return fused_lm_ce_forward_cuda(
      hidden.contiguous(),
      weight.contiguous(),
      labels.contiguous(),
      ignore_index);
}

std::vector<torch::Tensor> fused_rms_norm_forward(
    torch::Tensor input,
    torch::Tensor weight,
    double eps) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  TORCH_CHECK(input.dim() >= 2, "input must be at least 2D");
  TORCH_CHECK(weight.dim() == 1, "weight must be [D]");
  TORCH_CHECK(input.size(-1) == weight.size(0), "input last dim must match weight");
  TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "weight must be float32");
  return fused_rms_norm_forward_cuda(input.contiguous(), weight.contiguous(), eps);
}

std::vector<torch::Tensor> fused_rms_norm_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor inv_rms) {
  TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  TORCH_CHECK(inv_rms.is_cuda(), "inv_rms must be a CUDA tensor");
  TORCH_CHECK(grad_output.sizes() == input.sizes(), "grad_output shape must match input");
  TORCH_CHECK(input.dim() >= 2, "input must be at least 2D");
  TORCH_CHECK(weight.dim() == 1, "weight must be [D]");
  TORCH_CHECK(input.size(-1) == weight.size(0), "input last dim must match weight");
  TORCH_CHECK(inv_rms.numel() == input.numel() / input.size(-1), "inv_rms rows must match input rows");
  TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "weight must be float32");
  return fused_rms_norm_backward_cuda(
      grad_output.contiguous(),
      input.contiguous(),
      weight.contiguous(),
      inv_rms.contiguous());
}

std::vector<torch::Tensor> fused_state_softmax_matmul_forward(
    torch::Tensor scores,
    torch::Tensor values,
    torch::Tensor mask,
    bool zero_invalid) {
  TORCH_CHECK(scores.is_cuda(), "scores must be a CUDA tensor");
  TORCH_CHECK(values.is_cuda(), "values must be a CUDA tensor");
  TORCH_CHECK(scores.dim() == 3, "scores must be [B, T, S]");
  TORCH_CHECK(values.dim() == 3, "values must be [B, S, D]");
  TORCH_CHECK(scores.size(0) == values.size(0), "scores and values batch must match");
  TORCH_CHECK(scores.size(2) == values.size(1), "scores slots must match values slots");
  TORCH_CHECK(scores.scalar_type() == values.scalar_type(), "scores and values dtype must match");
  if (mask.numel() > 0) {
    TORCH_CHECK(mask.is_cuda(), "mask must be a CUDA tensor");
    TORCH_CHECK(mask.scalar_type() == torch::kBool, "mask must be bool");
    TORCH_CHECK(mask.dim() == 3, "mask must be [B or 1, T, S]");
    TORCH_CHECK(mask.size(0) == 1 || mask.size(0) == scores.size(0), "mask batch must be 1 or B");
    TORCH_CHECK(mask.size(1) == scores.size(1), "mask tokens must match scores");
    TORCH_CHECK(mask.size(2) == scores.size(2), "mask slots must match scores");
  }
  return fused_state_softmax_matmul_forward_cuda(
      scores.contiguous(),
      values.contiguous(),
      mask.numel() > 0 ? mask.contiguous() : mask,
      zero_invalid);
}

std::vector<torch::Tensor> fused_state_softmax_matmul_backward(
    torch::Tensor grad_context,
    torch::Tensor grad_weights,
    torch::Tensor weights,
    torch::Tensor values,
    torch::Tensor mask) {
  TORCH_CHECK(grad_context.is_cuda(), "grad_context must be a CUDA tensor");
  TORCH_CHECK(weights.is_cuda(), "weights must be a CUDA tensor");
  TORCH_CHECK(values.is_cuda(), "values must be a CUDA tensor");
  TORCH_CHECK(grad_context.dim() == 3, "grad_context must be [B, T, D]");
  TORCH_CHECK(weights.dim() == 3, "weights must be [B, T, S]");
  TORCH_CHECK(values.dim() == 3, "values must be [B, S, D]");
  TORCH_CHECK(grad_context.size(0) == weights.size(0), "grad_context and weights batch must match");
  TORCH_CHECK(grad_context.size(1) == weights.size(1), "grad_context and weights tokens must match");
  TORCH_CHECK(values.size(0) == weights.size(0), "values and weights batch must match");
  TORCH_CHECK(values.size(1) == weights.size(2), "values slots must match weights");
  TORCH_CHECK(values.size(2) == grad_context.size(2), "values dim must match grad_context");
  TORCH_CHECK(grad_context.scalar_type() == weights.scalar_type(), "grad_context and weights dtype must match");
  TORCH_CHECK(values.scalar_type() == weights.scalar_type(), "values and weights dtype must match");
  if (grad_weights.numel() > 0) {
    TORCH_CHECK(grad_weights.is_cuda(), "grad_weights must be a CUDA tensor");
    TORCH_CHECK(grad_weights.sizes() == weights.sizes(), "grad_weights shape must match weights");
    TORCH_CHECK(grad_weights.scalar_type() == weights.scalar_type(), "grad_weights dtype must match weights");
  }
  if (mask.numel() > 0) {
    TORCH_CHECK(mask.is_cuda(), "mask must be a CUDA tensor");
    TORCH_CHECK(mask.scalar_type() == torch::kBool, "mask must be bool");
    TORCH_CHECK(mask.sizes() == weights.sizes(), "expanded mask shape must match weights");
  }
  return fused_state_softmax_matmul_backward_cuda(
      grad_context.contiguous(),
      grad_weights.numel() > 0 ? grad_weights.contiguous() : grad_weights,
      weights.contiguous(),
      values.contiguous(),
      mask.numel() > 0 ? mask.contiguous() : mask);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_lm_ce_forward", &fused_lm_ce_forward, "Fused LM CE forward (CUDA)");
  m.def("fused_rms_norm_forward", &fused_rms_norm_forward, "Fused RMSNorm forward (CUDA)");
  m.def("fused_rms_norm_backward", &fused_rms_norm_backward, "Fused RMSNorm backward (CUDA)");
  m.def(
      "fused_state_softmax_matmul_forward",
      &fused_state_softmax_matmul_forward,
      "Fused state softmax-matmul forward (CUDA)");
  m.def(
      "fused_state_softmax_matmul_backward",
      &fused_state_softmax_matmul_backward,
      "Fused state softmax-matmul backward (CUDA)");
}
