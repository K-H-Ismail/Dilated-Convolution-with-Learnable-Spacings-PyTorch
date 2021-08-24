#include <torch/extension.h>

#include <iostream>

#include <vector>

// CUDA forward declarations

torch::Tensor sparse_weight_conv_cuda_forward(
    torch::Tensor input,    
    torch::Tensor weight,
    torch::Tensor bias,   
    const int dilation_h, const int dilation_w, 
    const int stride_h, const int stride_w, 
    const int padding_h, const int padding_w, 
    const int groups); 

std::vector<torch::Tensor> sparse_weight_conv_cuda_backward(
    torch::Tensor input,    
    torch::Tensor weight,
    torch::Tensor grad_output,      
    torch::Tensor bias,    
    const int dilation_h, const int dilation_w, 
    const int stride_h, const int stride_w, 
    const int padding_h, const int padding_w, 
    const int groups); 
    
// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor sparse_weight_conv_forward(
    torch::Tensor input,    
    torch::Tensor weight,
    torch::Tensor bias,    
    const int dilation_h, const int dilation_w, 
    const int stride_h, const int stride_w, 
    const int padding_h, const int padding_w, 
    const int groups) {

    CHECK_INPUT(input);    
    CHECK_CUDA(weight);
    CHECK_INPUT(bias);


    return sparse_weight_conv_cuda_forward(input, 
                                           weight, 
                                           bias,                                         
                                           dilation_h, dilation_w,
                                           stride_h, stride_w,
                                           padding_h, padding_w,
                                           groups); 
}

std::vector<torch::Tensor> sparse_weight_conv_backward(
    torch::Tensor input,    
    torch::Tensor weight,
    torch::Tensor grad_output,      
    torch::Tensor bias,    
    const int dilation_h, const int dilation_w, 
    const int stride_h, const int stride_w, 
    const int padding_h, const int padding_w, 
    const int groups) {
    
    CHECK_INPUT(input);    
    CHECK_CUDA(weight);
    CHECK_INPUT(grad_output);    
    CHECK_INPUT(bias);

    return sparse_weight_conv_cuda_backward(input, 
                                           weight, 
                                           grad_output,
                                           bias,                                           
                                           dilation_h, dilation_w,
                                           stride_h, stride_w,
                                           padding_h, padding_w,
                                           groups); 
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &sparse_weight_conv_forward, "Sparse_Weight_Conv forward (CUDA)");
  m.def("backward", &sparse_weight_conv_backward, "Sparse_Weight_Conv backward (CUDA)");
}