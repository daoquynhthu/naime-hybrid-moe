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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_lm_ce_forward", &fused_lm_ce_forward, "Fused LM CE forward (CUDA)");
  m.def("fused_rms_norm_forward", &fused_rms_norm_forward, "Fused RMSNorm forward (CUDA)");
  m.def("fused_rms_norm_backward", &fused_rms_norm_backward, "Fused RMSNorm backward (CUDA)");
}
