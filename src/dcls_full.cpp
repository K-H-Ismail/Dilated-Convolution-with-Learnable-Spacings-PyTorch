#include <torch/extension.h>

#include <iostream>

#include <vector>

// CUDA forward declarations

torch::Tensor dcls_full_cuda_forward(   
    torch::Tensor weight,
    torch::Tensor P1,
    torch::Tensor P2,
    const int dilation_h, const int dilation_w
    ); 

std::vector<torch::Tensor> dcls_full_cuda_backward(   
    torch::Tensor weight,
    torch::Tensor P1,
    torch::Tensor P2,
    torch::Tensor grad_output,      
    const int dilation_h, const int dilation_w
    ); 
    
// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor dcls_full_forward(   
    torch::Tensor weight,
    torch::Tensor P1,
    torch::Tensor P2,
    const int dilation_h, const int dilation_w
    ) {
  
    CHECK_INPUT(weight);
    CHECK_INPUT(P1);
    CHECK_INPUT(P2);


    return dcls_full_cuda_forward(
                              weight, 
                              P1, 
                              P2,
                              dilation_h, dilation_w
                              ); 
}

std::vector<torch::Tensor> dcls_full_backward(   
    torch::Tensor weight,
    torch::Tensor P1,
    torch::Tensor P2,
    torch::Tensor grad_output,      
    const int dilation_h, const int dilation_w
    ) {
      
    CHECK_INPUT(weight);
    CHECK_INPUT(P1);
    CHECK_INPUT(P2);
    CHECK_INPUT(grad_output);    

    return dcls_full_cuda_backward( 
                              weight, 
                              P1, 
                              P2,
                              grad_output,
                              dilation_h, dilation_w
                              ); 
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &dcls_full_forward, "DCLSFULL forward (CUDA)");
  m.def("backward", &dcls_full_backward, "DCLSFULL backward (CUDA)");
}