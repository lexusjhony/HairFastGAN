#include <torch/extension.h>

// Declaración de la función de operación
torch::Tensor fused_bias_act_op(const torch::Tensor& input, const torch::Tensor& bias, const torch::Tensor& refer,
    int act, int grad, float alpha, float scale);

// Macro para verificar si un tensor es un tensor CUDA
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
// Macro para verificar si un tensor es contiguo
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
// Macro para realizar ambas verificaciones
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Implementación de la función de fusión de bias y activación
torch::Tensor fused_bias_act(const torch::Tensor& input, const torch::Tensor& bias, const torch::Tensor& refer,
    int act, int grad, float alpha, float scale) {
    CHECK_INPUT(input);
    CHECK_INPUT(bias);
    if (refer.defined()) {
        CHECK_INPUT(refer);
    }

    return fused_bias_act_op(input, bias, refer, act, grad, alpha, scale);
}

// Definición del módulo de PyTorch
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_bias_act", &fused_bias_act, "fused bias act (CUDA)");
}
