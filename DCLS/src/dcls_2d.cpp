#include <torch/extension.h>

#include <iostream>

#include <vector>

// CUDA forward declarations

torch::Tensor dcls_2d_cuda_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor P1,
    torch::Tensor P2,
    c10::optional<torch::Tensor> bias,
    const int dilated_kernel_size_h, const int dilated_kernel_size_w,
    const int stride_h, const int stride_w,
    const int padding_h, const int padding_w,
    const int groups,
    const float scaling);

std::vector<torch::Tensor> dcls_2d_cuda_backward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor P1,
    torch::Tensor P2,
    torch::Tensor grad_output,
    c10::optional<torch::Tensor> bias,
    const int dilated_kernel_size_h, const int dilated_kernel_size_w,
    const int stride_h, const int stride_w,
    const int padding_h, const int padding_w,
    const int groups,
    const float scaling);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor dcls_2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor P1,
    torch::Tensor P2,
    c10::optional<torch::Tensor> bias,
    const int dilated_kernel_size_h, const int dilated_kernel_size_w,
    const int stride_h, const int stride_w,
    const int padding_h, const int padding_w,
    const int groups,
    const float scaling) {

    //CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(P1);
    CHECK_INPUT(P2);
    //CHECK_INPUT(bias);


    return dcls_2d_cuda_forward(input,
                              weight,
                              P1,
                              P2,
                              bias,
                              dilated_kernel_size_h, dilated_kernel_size_w,
                              stride_h, stride_w,
                              padding_h, padding_w,
                              groups,
                              scaling);
}

std::vector<torch::Tensor> dcls_2d_backward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor P1,
    torch::Tensor P2,
    torch::Tensor grad_output,
    c10::optional<torch::Tensor> bias,
    const int dilated_kernel_size_h, const int dilated_kernel_size_w,
    const int stride_h, const int stride_w,
    const int padding_h, const int padding_w,
    const int groups,
    const float scaling) {

    //CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(P1);
    CHECK_INPUT(P2);
    CHECK_INPUT(grad_output);
    //CHECK_INPUT(bias);

    return dcls_2d_cuda_backward(input,
                              weight,
                              P1,
                              P2,
                              grad_output,
                              bias,
                              dilated_kernel_size_h, dilated_kernel_size_w,
                              stride_h, stride_w,
                              padding_h, padding_w,
                              groups,
                              scaling);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &dcls_2d_forward, "DCLS forward (CUDA)");
  m.def("backward", &dcls_2d_backward, "DCLS backward (CUDA)");
}
