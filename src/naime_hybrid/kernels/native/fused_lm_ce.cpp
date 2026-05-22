#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> fused_lm_ce_forward_cuda(
    torch::Tensor hidden,
    torch::Tensor weight,
    torch::Tensor labels,
    int64_t ignore_index);

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_lm_ce_forward", &fused_lm_ce_forward, "Fused LM CE forward (CUDA)");
}
