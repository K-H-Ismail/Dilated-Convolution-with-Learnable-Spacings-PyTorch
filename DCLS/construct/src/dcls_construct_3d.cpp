#include <torch/extension.h>

#include <iostream>

#include <vector>

// CUDA forward declarations

torch::Tensor dcls_construct_3d_cuda_forward(   
    torch::Tensor weight,
    torch::Tensor P1,
    torch::Tensor P2,
    torch::Tensor P3,    
    const int dilation_d, const int dilation_h, const int dilation_w
    ); 

std::vector<torch::Tensor> dcls_construct_3d_cuda_backward(   
    torch::Tensor weight,
    torch::Tensor P1,
    torch::Tensor P2,
    torch::Tensor P3,    
    torch::Tensor grad_output,      
    const int dilation_d, const int dilation_h, const int dilation_w
    ); 
    
// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor dcls_construct_3d_forward(   
    torch::Tensor weight,
    torch::Tensor P1,
    torch::Tensor P2,
    torch::Tensor P3,
    const int dilation_d, const int dilation_h, const int dilation_w
    ) {
  
    CHECK_INPUT(weight);
    CHECK_INPUT(P1);
    CHECK_INPUT(P2);
    CHECK_INPUT(P3);    


    return dcls_construct_3d_cuda_forward(
                              weight, 
                              P1,
                              P2,
                              P3,                              
                              dilation_d,
                              dilation_h,
                              dilation_w                              
                              ); 
}

std::vector<torch::Tensor> dcls_construct_3d_backward(   
    torch::Tensor weight,
    torch::Tensor P1,
    torch::Tensor P2,
    torch::Tensor P3,
    torch::Tensor grad_output,      
    const int dilation_d, const int dilation_h, const int dilation_w
    ) {
      
    CHECK_INPUT(weight);
    CHECK_INPUT(P1);
    CHECK_INPUT(P2);
    CHECK_INPUT(P3);
    CHECK_INPUT(grad_output);    

    return dcls_construct_3d_cuda_backward( 
                              weight, 
                              P1,
                              P2,
                              P3,
                              grad_output,
                              dilation_d,
                              dilation_h,
                              dilation_w 
                              ); 
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &dcls_construct_3d_forward, "DCLS3d forward (CUDA)");
  m.def("backward", &dcls_construct_3d_backward, "DCLS3d backward (CUDA)");
}