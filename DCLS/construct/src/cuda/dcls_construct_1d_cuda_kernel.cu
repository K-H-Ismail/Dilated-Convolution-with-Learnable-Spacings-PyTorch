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
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> P,   
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> W1, 
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> W2,   
    const int ch_out, const int ch_in,
    const int kernel,
    const int length_out,    
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> interpolated_weight) {
  CUDA_KERNEL_LOOP(index, n) {
    int l_out = index % kernel;
    int channel_in = (index / kernel) % ch_in;
    int channel_out = (index / kernel / ch_in) % ch_out;
      
    int p = P[channel_out][channel_in][l_out];
    int p_next = p + 1;
      
    if(p >= 0 & p < length_out)
    {   
        interpolated_weight[channel_out][channel_in][p] += W1[channel_out][channel_in][l_out];
        if(p_next < length_out) 
            interpolated_weight[channel_out][channel_in][p_next] += W2[channel_out][channel_in][l_out]; 
    }
  }
}

template <typename scalar_t>
__global__ void interpolation_grad_kernel(
    const int n,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_output,    
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> P, 
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> W1, 
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> W2,
    const int ch_out, const int ch_in,
    const int kernel,
    const int length_out,   
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits>  interpolated_weight) {
  CUDA_KERNEL_LOOP(index, n) {
    int l_out = index % kernel;
    int channel_in = (index / kernel) % ch_in;
    int channel_out = (index / kernel / ch_in) % ch_out;
      
    int p = P[channel_out][channel_in][l_out];

    int p_next = p + 1;  
      
    if(p >= 0 & p < length_out)
    {  
        interpolated_weight[channel_out][channel_in][l_out] += 
            grad_output[channel_out][channel_in][p] * W1[channel_out][channel_in][l_out];

        if(p_next < length_out)
            interpolated_weight[channel_out][channel_in][l_out] +=        
            grad_output[channel_out][channel_in][p_next] * W2[channel_out][channel_in][l_out];
    }
      
  }
}

torch::Tensor  dcls_construct_1d_cuda_forward(  
    torch::Tensor weight,
    torch::Tensor P1,
    const int dilation
    ) {
    
    const int channels_out = weight.size(0);
    const int channels_in = weight.size(1);    
    const int kernel = weight.size(2);
 
    const int half_range_bot = dilation*kernel/2;
    
    // Suitable for Kaiming uniform initialization
    auto scaling = sqrt(kernel * channels_in * dilation * dilation / 4);    

    auto scaled_P = P1*scaling + at::arange(-half_range_bot + dilation/2,half_range_bot + 1e-7,dilation, weight.options())
                            .repeat({channels_out,channels_in,1});
                            
    auto P = scaled_P.floor();
    auto rest = scaled_P - P;
    
    const int length_out = dilation * kernel + (dilation+1)%2;
    
    P += dilation*kernel/2 ;
    P = P.clamp(0,length_out-1);

    auto W2 = rest * weight;     
    auto W1 = weight - W2;  
    
    auto output = torch::zeros({channels_out, channels_in, length_out}, weight.options());
    
    const int num_kernels =  channels_out * channels_in * kernel;
    AT_DISPATCH_FLOATING_TYPES(weight.type(), "dcls_construct_1d_forward_cuda", [&] {
          
        interpolation_kernel<scalar_t><<<GET_BLOCKS(num_kernels), 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                     num_kernels,
                                     P.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     W1.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     W2.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     channels_out, channels_in,
                                     kernel,
                                     length_out,
                                     output.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
    });    
    return output;
}

std::vector<torch::Tensor> dcls_construct_1d_cuda_backward(   
    torch::Tensor weight,
    torch::Tensor P1,
    torch::Tensor grad_output,      
    const int dilation
    ) {
    
    auto grad_weight = torch::zeros_like(weight);
    auto grad_P1 = torch::zeros_like(P1);    
        
    const int channels_out = weight.size(0);
    const int channels_in = weight.size(1);    
    const int kernel = weight.size(2);
    
    const int half_range_bot = dilation*kernel/2;
    const int half_range_top = half_range_bot - (dilation*kernel+1)%2;
    
    // Suitable for Kaiming uniform initialization
    auto scaling = sqrt(kernel * channels_in * dilation * dilation / 4);     

    auto scaled_P = P1*scaling + at::arange(-half_range_bot + dilation/2,half_range_bot + 1e-7,dilation, weight.options())
                            .repeat({channels_out,channels_in,1});
                            
    auto P = scaled_P.floor();
    auto rest = scaled_P - P;
    
    const int length_out = dilation * kernel + (dilation+1)%2;    
    
    P += dilation*kernel/2 ;
    P = P.clamp(0,length_out-1);  

    auto ones = at::ones_like(rest, weight.options());
    auto W2 = rest;     
    auto W1 = ones - W2; 
    
    auto sigma = 0.5*ones;    
    
    auto W1_P = d_floor(scaled_P, sigma, half_range_bot, half_range_top, d_zero()) * weight - weight;
    auto W2_P = - W1_P;
    

    const int num_kernels = channels_out * channels_in * kernel;    
    AT_DISPATCH_FLOATING_TYPES(weight.type(), "dcls_construct_1d_backward_cuda", [&] {
             
        interpolation_grad_kernel<scalar_t><<<GET_BLOCKS(num_kernels), 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                     num_kernels,
                                     grad_output.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     P.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     W1.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     W2.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     channels_out, channels_in,
                                     kernel, 
                                     length_out,                                 
                                     grad_weight.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
        interpolation_grad_kernel<scalar_t><<<GET_BLOCKS(num_kernels), 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                     num_kernels,
                                     grad_output.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     P.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     W1_P.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     W2_P.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     channels_out, channels_in,
                                     kernel, 
                                     length_out,                                     
                                     grad_P1.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
        
    });

    return {grad_weight,
            grad_P1*scaling};
}
