#include <iostream>
#include <torch/extension.h>

torch::Tensor im2col_dcls_2d_batch_cuda(torch::Tensor im,
                               torch::Tensor P_h, torch::Tensor P_w,                      
                               const int dilation_h, const int dilation_w,
                               const int padding_h, const int padding_w,
                               const int stride_h, const int stride_w,
                               const int height_out, const int width_out                            
                               );
// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor im2col_dcls_2d_forward(torch::Tensor im,
                          torch::Tensor P_h, torch::Tensor P_w,                      
                          const int dilation_h, const int dilation_w,
                          const int padding_h, const int padding_w,
                          const int stride_h, const int stride_w,
                          const int height_out, const int width_out,                                  
                          const int shift_h, const int shift_w                                    
                          )
{   
    CHECK_INPUT(im);
    CHECK_INPUT(P_h);
    CHECK_INPUT(P_w);    

    return im2col_dcls_2d_batch_cuda(im,
                            P_h, P_w,    
                            dilation_h, dilation_w,
                            padding_h, padding_w,                              
                            stride_h, stride_w,
                            height_out, width_out
                            ); 
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &im2col_dcls_2d_forward);
}