#include <torch/extension.h>

#include <iostream>

#include <vector>

// CUDA forward declarations

torch::Tensor dcls_construct_2d_cuda_forward(   
    torch::Tensor weight,
    torch::Tensor P1,
    torch::Tensor P2,
    const int dilation_h, const int dilation_w,
    const float gain
    ); 

std::vector<torch::Tensor> dcls_construct_2d_cuda_backward(   
    torch::Tensor weight,
    torch::Tensor P1,
    torch::Tensor P2,
    torch::Tensor grad_output,      
    const int dilation_h, const int dilation_w,
    const float gain
    ); 
    
// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor dcls_construct_2d_forward(   
    torch::Tensor weight,
    torch::Tensor P1,
    torch::Tensor P2,
    const int dilation_h, const int dilation_w,
    const float gain
    ) {
  
    CHECK_INPUT(weight);
    CHECK_INPUT(P1);
    CHECK_INPUT(P2);


    return dcls_construct_2d_cuda_forward(
                              weight, 
                              P1, 
                              P2,
                              dilation_h, dilation_w,
                              gain
                              ); 
}

std::vector<torch::Tensor> dcls_construct_2d_backward(   
    torch::Tensor weight,
    torch::Tensor P1,
    torch::Tensor P2,
    torch::Tensor grad_output,      
    const int dilation_h, const int dilation_w,
    const float gain
    ) {
      
    CHECK_INPUT(weight);
    CHECK_INPUT(P1);
    CHECK_INPUT(P2);
    CHECK_INPUT(grad_output);    

    return dcls_construct_2d_cuda_backward( 
                              weight, 
                              P1, 
                              P2,
                              grad_output,
                              dilation_h, dilation_w,
                              gain
                              ); 
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &dcls_construct_2d_forward, "DCLS2d forward (CUDA)");
  m.def("backward", &dcls_construct_2d_backward, "DCLS2d backward (CUDA)");
}