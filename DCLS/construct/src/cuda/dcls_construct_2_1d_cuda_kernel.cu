#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include "dcls.h"

#include <math.h>
#include <vector>

template <typename scalar_t>
__global__ void interpolation_kernel(
    const int n,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> P,   
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> W1, 
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> W2,   
    const int ch_out, const int ch_in,
    const int kernel_h, const int kernel_w,    
    const int height_out,    
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> interpolated_weight) {
  CUDA_KERNEL_LOOP(index, n) {
    int w_out = index % kernel_w;
    int h_out = (index / kernel_w) % kernel_h;
    int channel_in = (index / kernel_h / kernel_w) % ch_in;
    int channel_out = (index / kernel_h / kernel_w / ch_in) % ch_out;
      
    int p = P[channel_out][channel_in][h_out][w_out];
    int p_next = p + 1;     
   
    if(p >= 0 & p < height_out)
    {         
        interpolated_weight[channel_out][channel_in][p][w_out] += W1[channel_out][channel_in][h_out][w_out];
        if(p_next < height_out) 
            interpolated_weight[channel_out][channel_in][p_next][w_out] += 
            W2[channel_out][channel_in][h_out][w_out];
    }
  }
}

template <typename scalar_t>
__global__ void interpolation_grad_kernel(
    const int n,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_output,    
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> P, 
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> W1, 
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> W2,
    const int ch_out, const int ch_in,
    const int kernel_h, const int kernel_w,    
    const int height_out,  
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits>  interpolated_weight) {
  CUDA_KERNEL_LOOP(index, n) {
    int w_out = index % kernel_w;
    int h_out = (index / kernel_w) % kernel_h;
    int channel_in = (index / kernel_h / kernel_w) % ch_in;
    int channel_out = (index / kernel_h / kernel_w / ch_in) % ch_out;
      
    int p = P[channel_out][channel_in][h_out][w_out];
    int p_next = p + 1;       
      
    if(p >= 0 & p < height_out)
    {       
        interpolated_weight[channel_out][channel_in][h_out][w_out] += 
            grad_output[channel_out][channel_in][p][w_out] * W1[channel_out][channel_in][h_out][w_out];

        if(p_next < height_out)
            interpolated_weight[channel_out][channel_in][h_out][w_out] += 
            grad_output[channel_out][channel_in][p_next][w_out] * W2[channel_out][channel_in][h_out][w_out];
    }
      
  }
}

torch::Tensor  dcls_construct_2_1d_cuda_forward(  
    torch::Tensor weight,
    torch::Tensor P1,
    const int dilation_h   
    ) {
    
    const int channels_out = weight.size(0);
    const int channels_in = weight.size(1);    
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);    
 
    const int half_range_bot = dilation_h*kernel_h/2;
    
    // Suitable for Kaiming uniform initialization
    auto scaling = sqrt(kernel_h * kernel_w * channels_in * dilation_h * dilation_h / 4);      

    auto scaled_P1 = P1*scaling + at::arange(-half_range_bot + dilation_h/2,half_range_bot,dilation_h, weight.options())
                            .repeat({kernel_w,1})
                            .t()
                            .repeat({channels_out,channels_in,1,1});
                            
    auto P = scaled_P1.floor();
    auto rest = scaled_P1 - P1;
    
    const int height_out = dilation_h * kernel_h + (dilation_h+1)%2;    
    
    P += dilation_h*kernel_h/2 ;
    P = P.clamp(0,height_out-1);     

    auto W2 = rest * weight;     
    auto W1 = weight - W2;
 
    const int width_out = kernel_w;
   
    auto output = torch::zeros({channels_out, channels_in, height_out, width_out}, weight.options());
    
    const int num_kernels =  channels_out * channels_in * kernel_h * kernel_w;
    AT_DISPATCH_FLOATING_TYPES(weight.type(), "dcls_construct_2_1d_forward_cuda", [&] {
          
        interpolation_kernel<scalar_t><<<GET_BLOCKS(num_kernels), 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                     num_kernels,
                                     P.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                     W1.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                     W2.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                     channels_out, channels_in,
                                     kernel_h, kernel_w,
                                     height_out,
                                     output.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
    });    
    return output;
}

std::vector<torch::Tensor> dcls_construct_2_1d_cuda_backward(   
    torch::Tensor weight,
    torch::Tensor P1,
    torch::Tensor grad_output,      
    const int dilation_h
    ) {
    
    auto grad_weight = torch::zeros_like(weight);
    auto grad_P1 = torch::zeros_like(P1);    
        
    const int channels_out = weight.size(0);
    const int channels_in = weight.size(1);    
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);
    
    const int half_range_bot = dilation_h*kernel_h/2;
    const int half_range_top = half_range_bot - (dilation_h*kernel_h + 1)%2;
    
    // Suitable for Kaiming uniform initialization
    auto scaling = sqrt(kernel_h * kernel_w * channels_in * dilation_h * dilation_h / 4);      
    
    auto scaled_P1 = P1*scaling + at::arange(-half_range_bot + dilation_h/2,half_range_bot,dilation_h, weight.options())
                            .repeat({kernel_w,1})
                            .t()
                            .repeat({channels_out,channels_in,1,1});
                            
    auto P = scaled_P1.floor();
    auto rest = scaled_P1 - P1;
    
    const int height_out = dilation_h * kernel_h + (dilation_h+1)%2;    
    
    P += dilation_h*kernel_h/2 ;
    P = P.clamp(0,height_out-1);     
    
    auto ones = at::ones_like(rest, weight.options());
    auto W2 = rest;     
    auto W1 = ones - W2; 
    
    auto sigma = 0.5*ones;   
    
    auto W1_P = d_floor(scaled_P1, sigma, half_range_bot, half_range_top, d_zero()) * weight - weight;
    auto W2_P = - W1_P;
    

    const int num_kernels =  channels_out * channels_in * kernel_h * kernel_w;    
    AT_DISPATCH_FLOATING_TYPES(weight.type(), "dcls_construct_2_1d_backward_cuda", [&] {
             
        interpolation_grad_kernel<scalar_t><<<GET_BLOCKS(num_kernels), 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                     num_kernels,
                                     grad_output.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                     P.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                     W1.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                     W2.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                     channels_out, channels_in,
                                     kernel_h, kernel_w,
                                     height_out,                                
                                     grad_weight.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
        interpolation_grad_kernel<scalar_t><<<GET_BLOCKS(num_kernels), 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                     num_kernels,
                                     grad_output.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                     P.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                     W1_P.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                     W2_P.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                     channels_out, channels_in,
                                     kernel_h, kernel_w,
                                     height_out,                                    
                                     grad_P1.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
        
    });


    return {grad_weight,
            grad_P1*scaling};
}
