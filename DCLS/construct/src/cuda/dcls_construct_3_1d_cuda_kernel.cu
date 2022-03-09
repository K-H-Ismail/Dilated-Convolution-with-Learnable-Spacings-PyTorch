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
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> P,   
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> W1, 
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> W2,   
    const int ch_out, const int ch_in,
    const int kernel_d, const int kernel_h, const int kernel_w,    
    const int depth_out,    
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> interpolated_weight) {
  CUDA_KERNEL_LOOP(index, n) {
    int w_out = index % kernel_w;
    int h_out = (index / kernel_w) % kernel_h;
    int d_out = ((index / kernel_w) / kernel_h) % kernel_d;      
    int channel_in = (index / kernel_d / kernel_h / kernel_w) % ch_in;
    int channel_out = (index / kernel_d / kernel_h / kernel_w / ch_in) % ch_out;
      
    int p = P[channel_out][channel_in][d_out][h_out][w_out];
    int p_next = p + 1;     
   
    if(p >= 0 & p < depth_out)
    {         
        interpolated_weight[channel_out][channel_in][p][h_out][w_out] += W1[channel_out][channel_in][d_out][h_out][w_out];
        if(p_next < depth_out) 
            interpolated_weight[channel_out][channel_in][p_next][h_out][w_out] += 
            W2[channel_out][channel_in][d_out][h_out][w_out];
    }
  }
}

template <typename scalar_t>
__global__ void interpolation_grad_kernel(
    const int n,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_output,    
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> P, 
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> W1, 
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> W2,
    const int ch_out, const int ch_in,
    const int kernel_d, const int kernel_h, const int kernel_w,    
    const int depth_out,  
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits>  interpolated_weight) {
  CUDA_KERNEL_LOOP(index, n) {
    int w_out = index % kernel_w;
    int h_out = (index / kernel_w) % kernel_h;
    int d_out = ((index / kernel_w) / kernel_h) % kernel_d;      
    int channel_in = (index / kernel_d / kernel_h / kernel_w) % ch_in;
    int channel_out = (index / kernel_d / kernel_h / kernel_w / ch_in) % ch_out;
      
    int p = P[channel_out][channel_in][d_out][h_out][w_out];
    int p_next = p + 1;       
      
    if(p >= 0 & p < depth_out)
    {       
        interpolated_weight[channel_out][channel_in][d_out][h_out][w_out] += 
            grad_output[channel_out][channel_in][p][h_out][w_out] * W1[channel_out][channel_in][d_out][h_out][w_out];

        if(p_next < depth_out)
            interpolated_weight[channel_out][channel_in][d_out][h_out][w_out] += 
            grad_output[channel_out][channel_in][p_next][h_out][w_out] * W2[channel_out][channel_in][d_out][h_out][w_out];
    }
      
  }
}

torch::Tensor  dcls_construct_3_1d_cuda_forward(  
    torch::Tensor weight,
    torch::Tensor P1,
    const int dilation_d,
    const float gain   
    ) {
    
    const int channels_out = weight.size(0);
    const int channels_in = weight.size(1);    
    const int kernel_d = weight.size(2);
    const int kernel_h = weight.size(3);
    const int kernel_w = weight.size(4);    
 
    const int half_range_d = (dilation_d * kernel_d) / 2;
    
    // Suitable for Kaiming uniform initialization
    auto scaling = gain * sqrt(kernel_d * kernel_h * kernel_w * channels_in * channels_out);     

    auto scaled_P1 = P1*scaling /*+ at::arange(-half_range_bot + dilation_d/2,half_range_bot,dilation_d, weight.options())
                            .repeat({kernel_h,kernel_w,1})
                            .permute({2,0,1})
                            .repeat({channels_out,channels_in,1,1,1})*/;
                            
    // Add d.k/2, positions are now uniformly around 0 and d.k - 1    
    auto P_d = scaled_P1 + half_range_d;
    
    // Apply floor function, positions are now integers uniformly around 0 and d.k - 1
    P_d = P_d.floor();
    
    // Apply clamp function, positions are now integers strictly between 0 and d.k - 1
    P_d = P_d.clamp(0, dilation_d * kernel_d - 1); 
    
    
    // Calculate rests for interpolation
    auto rest_d = (scaled_P1 + half_range_d).clamp(0, dilation_d * kernel_d - 1) - P_d;     
    
    const int depth_out = dilation_d * kernel_d;
    const int height_out = kernel_h;
    const int width_out = kernel_w;
    
    auto W2 = rest_d * weight;     
    auto W1 = weight - W2;    
   
    auto output = torch::zeros({channels_out, channels_in, depth_out, height_out, width_out}, weight.options());
    
    const int num_kernels =  channels_out * channels_in * kernel_d * kernel_h * kernel_w;
    AT_DISPATCH_FLOATING_TYPES(weight.type(), "dcls_construct_3_1d_forward_cuda", [&] {
          
        interpolation_kernel<scalar_t><<<GET_BLOCKS(num_kernels), 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                     num_kernels,
                                     P_d.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W1.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W2.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     channels_out, channels_in,
                                     kernel_d, kernel_h, kernel_w,
                                     depth_out,
                                     output.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>());
    });    
    return output;
}

std::vector<torch::Tensor> dcls_construct_3_1d_cuda_backward(   
    torch::Tensor weight,
    torch::Tensor P1,
    torch::Tensor grad_output,      
    const int dilation_d,
    const float gain
    ) {
    
    auto grad_weight = torch::zeros_like(weight);
    auto grad_P1 = torch::zeros_like(P1);    
        
    const int channels_out = weight.size(0);
    const int channels_in = weight.size(1);    
    const int kernel_d = weight.size(2);
    const int kernel_h = weight.size(3);
    const int kernel_w = weight.size(4);
    
    const int half_range_d = (dilation_d * kernel_d) / 2;
    
    // Suitable for Kaiming uniform initialization
    auto scaling = gain * sqrt(kernel_d * kernel_h * kernel_w * channels_in * channels_out);     
    
    auto scaled_P1 = P1*scaling /*+ at::arange(-half_range_bot + dilation_d/2,half_range_bot,dilation_d, weight.options())
                            .repeat({kernel_h,kernel_w,1})
                            .permute({2,0,1})
                            .repeat({channels_out,channels_in,1,1,1})*/;
                            
    // Add d.k/2, positions are now uniformly around 0 and d.k - 1    
    auto P_d = scaled_P1 + half_range_d;
    
    // Apply floor function, positions are now integers uniformly around 0 and d.k - 1
    P_d = P_d.floor();
    
    // Apply clamp function, positions are now integers strictly between 0 and d.k - 1
    P_d = P_d.clamp(0, dilation_d * kernel_d - 1); 
    
    
    // Calculate $s for interpolation
    
    const int depth_out = dilation_d;
        
    // Calculate rests for interpolation
    
    auto rest_d = scaled_P1 + half_range_d;
    auto mask_d = rest_d.ge(0) * rest_d.le(depth_out-1);
    rest_d = rest_d.clamp(0,depth_out-1) - P_d;  
    
         
    auto W2 = rest_d * weight;     
    auto W1 = weight - W2;    
    
    auto W1_P = weight * mask_d;
    auto W2_P = -W1_P;


    const int num_kernels =  channels_out * channels_in * kernel_d * kernel_h * kernel_w;    
    AT_DISPATCH_FLOATING_TYPES(weight.type(), "dcls_construct_3_1d_backward_cuda", [&] {
             
        interpolation_grad_kernel<scalar_t><<<GET_BLOCKS(num_kernels), 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                     num_kernels,
                                     grad_output.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     P_d.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W1.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W2.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     channels_out, channels_in,
                                     kernel_d, kernel_h, kernel_w,
                                     depth_out,                                
                                     grad_weight.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>());
        interpolation_grad_kernel<scalar_t><<<GET_BLOCKS(num_kernels), 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                     num_kernels,
                                     grad_output.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     P_d.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W1_P.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W2_P.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     channels_out, channels_in,
                                     kernel_d, kernel_h, kernel_w,
                                     depth_out,                                    
                                     grad_P1.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>());
        
    });


    return {grad_weight,
            grad_P1*scaling};
}
