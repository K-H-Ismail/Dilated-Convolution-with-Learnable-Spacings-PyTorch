#include <torch/extension.h>

#include <iostream>

#include <vector>

// CUDA forward declarations

torch::Tensor dcls_construct_2_1d_cuda_forward(   
    torch::Tensor weight,
    torch::Tensor P,
    const int dilation
    ); 

std::vector<torch::Tensor> dcls_construct_2_1d_cuda_backward(   
    torch::Tensor weight,
    torch::Tensor P,
    torch::Tensor grad_output,      
    const int dilation
    ); 
    
// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor dcls_construct_2_1d_forward(   
    torch::Tensor weight,
    torch::Tensor P,
    const int dilation
    ) {
  
    CHECK_INPUT(weight);
    CHECK_INPUT(P);


    return dcls_construct_2_1d_cuda_forward(
                              weight, 
                              P,
                              dilation
                              ); 
}

std::vector<torch::Tensor> dcls_construct_2_1d_backward(   
    torch::Tensor weight,
    torch::Tensor P,
    torch::Tensor grad_output,      
    const int dilation
    ) {
      
    CHECK_INPUT(weight);
    CHECK_INPUT(P);
    CHECK_INPUT(grad_output);    

    return dcls_construct_2_1d_cuda_backward( 
                              weight, 
                              P,
                              grad_output,
                              dilation
                              ); 
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &dcls_construct_2_1d_forward, "DCLS2_1d forward (CUDA)");
  m.def("backward", &dcls_construct_2_1d_backward, "DCLS2_1d backward (CUDA)");
}