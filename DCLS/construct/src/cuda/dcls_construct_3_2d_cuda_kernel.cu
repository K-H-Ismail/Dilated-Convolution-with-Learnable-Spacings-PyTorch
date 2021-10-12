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
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> P_d,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> P_h,    
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> W1, 
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> W2,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> W3, 
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> W4,     
    const int ch_out, const int ch_in,
    const int kernel_d, const int kernel_h, const int kernel_w,    
    const int depth_out, const int height_out,    
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> interpolated_weight) {
  CUDA_KERNEL_LOOP(index, n) {
    int w_out = index % kernel_w;
    int h_out = (index / kernel_w) % kernel_h;
    int d_out = ((index / kernel_w) / kernel_h) % kernel_d;      
    int channel_in = (index / kernel_d / kernel_h / kernel_w) % ch_in;
    int channel_out = (index / kernel_d / kernel_h / kernel_w / ch_in) % ch_out;
      
    int p_d = P_d[channel_out][channel_in][d_out][h_out][w_out];
    int p_h = P_h[channel_out][channel_in][d_out][h_out][w_out];
    int p_d_next = p_d + 1;
    int p_h_next = p_h + 1;  
   
    if(p_d >= 0 & p_d < depth_out & p_h >= 0 & p_h < height_out)
    {   
        interpolated_weight[channel_out][channel_in][p_d][p_h][w_out] +=  W1[channel_out][channel_in][d_out][h_out][w_out];
        if(p_d_next < depth_out) 
            interpolated_weight[channel_out][channel_in][p_d_next][p_h][w_out] += W2[channel_out][channel_in][d_out][h_out][w_out];
        if(p_h_next < height_out) 
            interpolated_weight[channel_out][channel_in][p_d][p_h_next][w_out] += W3[channel_out][channel_in][d_out][h_out][w_out];
        if(p_d_next < depth_out & p_h_next < height_out) 
            interpolated_weight[channel_out][channel_in][p_d_next][p_h_next][w_out] += W4[channel_out][channel_in][d_out][h_out][w_out];
    }
  }
}

template <typename scalar_t>
__global__ void interpolation_grad_kernel(
    const int n,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_output,    
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> P_d,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> P_h,    
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> W1, 
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> W2,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> W3, 
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> W4,     
    const int ch_out, const int ch_in,
    const int kernel_d, const int kernel_h, const int kernel_w,    
    const int depth_out, const int height_out,    
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> interpolated_weight) {
  CUDA_KERNEL_LOOP(index, n) {
    int w_out = index % kernel_w;
    int h_out = (index / kernel_w) % kernel_h;
    int d_out = ((index / kernel_w) / kernel_h) % kernel_d;      
    int channel_in = (index / kernel_d / kernel_h / kernel_w) % ch_in;
    int channel_out = (index / kernel_d / kernel_h / kernel_w / ch_in) % ch_out;
      
    int p_d = P_d[channel_out][channel_in][d_out][h_out][w_out];
    int p_h = P_h[channel_out][channel_in][d_out][h_out][w_out];
    int p_d_next = p_d + 1;
    int p_h_next = p_h + 1;       
      
    if(p_d >= 0 & p_d < depth_out & p_h >= 0 & p_h < height_out)
    {   
        interpolated_weight[channel_out][channel_in][d_out][h_out][w_out] += grad_output[channel_out][channel_in][p_d][p_h][w_out] * W1[channel_out][channel_in][d_out][h_out][w_out];
        if(p_d_next < depth_out) 
            interpolated_weight[channel_out][channel_in][d_out][h_out][w_out] += grad_output[channel_out][channel_in][p_d_next][p_h][w_out] * W2[channel_out][channel_in][d_out][h_out][w_out];
        if(p_h_next < height_out) 
            interpolated_weight[channel_out][channel_in][d_out][h_out][w_out] += grad_output[channel_out][channel_in][p_d][p_h_next][w_out] * W3[channel_out][channel_in][d_out][h_out][w_out];
        if(p_d_next < depth_out & p_h_next < height_out) 
            interpolated_weight[channel_out][channel_in][d_out][h_out][w_out] += grad_output[channel_out][channel_in][p_d_next][p_h_next][w_out] * W4[channel_out][channel_in][d_out][h_out][w_out];
    }
      
  }
}

torch::Tensor  dcls_construct_3_2d_cuda_forward(  
    torch::Tensor weight,
    torch::Tensor P1,
    torch::Tensor P2,    
    const int dilation_d, const int dilation_h   
    ) {
    
    const int channels_out = weight.size(0);
    const int channels_in = weight.size(1);    
    const int kernel_d = weight.size(2);
    const int kernel_h = weight.size(3);
    const int kernel_w = weight.size(4);    

    const int half_range_bot_d = dilation_d*kernel_d/2;
    const int half_range_bot_h = dilation_h*kernel_h/2;
    
    // Suitable for Kaiming uniform initialization
    auto scaling_d = sqrt(kernel_h * kernel_w * kernel_d * channels_in * dilation_d * dilation_d / 4);
    auto scaling_h = sqrt(kernel_h * kernel_w * kernel_d * channels_in * dilation_h * dilation_h / 4);    
    
    auto scaled_P1 = P1*scaling_d + at::arange(-half_range_bot_d + dilation_d/2,half_range_bot_d + 1e-7,dilation_d, weight.options())
                            .repeat({kernel_h,kernel_w,1})
                            .permute({2,0,1})
                            .repeat({channels_out,channels_in,1,1,1});
    auto scaled_P2 = P2*scaling_h + at::arange(-half_range_bot_h + dilation_h/2,half_range_bot_h + 1e-7,dilation_h, weight.options())
                            .repeat({kernel_d,kernel_w,1})
                            .permute({0,2,1})
                            .repeat({channels_out,channels_in,1,1,1});
                            
    auto P_d = scaled_P1.floor();
    auto rest_d = scaled_P1 - P_d;
    
    auto P_h = scaled_P2.floor();
    auto rest_h = scaled_P2 - P_h;

    const int depth_out = dilation_d * kernel_d + (dilation_d+1)%2;
    const int height_out = dilation_h * kernel_h + (dilation_h+1)%2;
    const int width_out = kernel_w;    

    P_d += dilation_d*kernel_d/2 ;
    P_h += dilation_h*kernel_h/2 ;
    P_d = P_d.clamp(0,depth_out-1); 
    P_h = P_h.clamp(0,height_out-1); 
    
    auto rdW = rest_d * weight;
    auto rhW = rest_h * weight;
    auto rdhW = rest_d * rhW;    
    auto W1 = weight - rdW - rhW + rdhW;
    auto W2 = rdW - rdhW;
    auto W3 = rhW - rdhW;
    auto W4 = rdhW; 
 
   
    auto output = torch::zeros({channels_out, channels_in, depth_out, height_out, width_out}, weight.options());
    
    const int num_kernels =  channels_out * channels_in * kernel_d * kernel_h * kernel_w;
    AT_DISPATCH_FLOATING_TYPES(weight.type(), "dcls_construct_3_2d_forward_cuda", [&] {
          
        interpolation_kernel<scalar_t><<<GET_BLOCKS(num_kernels), 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                     num_kernels,
                                     P_d.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     P_h.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W1.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W2.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W3.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W4.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     channels_out, channels_in,
                                     kernel_d, kernel_h, kernel_w,
                                     depth_out, height_out,
                                     output.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>());
    });    
    return output;
}

std::vector<torch::Tensor> dcls_construct_3_2d_cuda_backward(   
    torch::Tensor weight,
    torch::Tensor P1,
    torch::Tensor P2,
    torch::Tensor grad_output,    
    const int dilation_d, const int dilation_h  
    ) {
    
    auto grad_weight = torch::zeros_like(weight);
    auto grad_P1 = torch::zeros_like(P1);
    auto grad_P2 = torch::zeros_like(P2);    
        
    const int channels_out = weight.size(0);
    const int channels_in = weight.size(1);    
    const int kernel_d = weight.size(2);
    const int kernel_h = weight.size(3);
    const int kernel_w = weight.size(4);
    
    
    const int half_range_bot_d = dilation_d*kernel_d/2;
    const int half_range_bot_h = dilation_h*kernel_h/2;
    
    const int half_range_top_d = half_range_bot_d - (dilation_d*kernel_d + 1)%2;
    const int half_range_top_h = half_range_bot_h - (dilation_h*kernel_h + 1)%2; 

    // Suitable for Kaiming uniform initialization
    auto scaling_d = sqrt(kernel_h * kernel_w * kernel_d * channels_in * dilation_d * dilation_d / 4);
    auto scaling_h = sqrt(kernel_h * kernel_w * kernel_d * channels_in * dilation_h * dilation_h / 4); 
    
    auto scaled_P1 = P1*scaling_d + at::arange(-half_range_bot_d + dilation_d/2,half_range_bot_d + 1e-7,dilation_d, weight.options())
                            .repeat({kernel_h,kernel_w,1})
                            .permute({2,0,1})
                            .repeat({channels_out,channels_in,1,1,1});
    auto scaled_P2 = P2*scaling_h + at::arange(-half_range_bot_h + dilation_h/2,half_range_bot_h + 1e-7,dilation_h, weight.options())
                            .repeat({kernel_d,kernel_w,1})
                            .permute({0,2,1})
                            .repeat({channels_out,channels_in,1,1,1});
                            
    auto P_d = scaled_P1.floor();
    auto rest_d = scaled_P1 - P_d;
    
    auto P_h = scaled_P2.floor();
    auto rest_h = scaled_P2 - P_h;

    const int depth_out = dilation_d * kernel_d + (dilation_d+1)%2;
    const int height_out = dilation_h * kernel_h + (dilation_h+1)%2;

    P_d += dilation_d*kernel_d/2 ;
    P_h += dilation_h*kernel_h/2 ;
    P_d = P_d.clamp(0,depth_out-1); 
    P_h = P_h.clamp(0,height_out-1); 
    
    auto rdW = rest_d * weight;
    auto rhW = rest_h * weight;
    auto rdhW = rest_d * rhW;
    auto rdh = rest_d * rest_h;
    
    auto ones = at::ones_like(rest_h, weight.options());
    auto sigma = 0.5*ones; 
    
    auto W1 = ones - rest_d - rest_h + rdh;
    auto W2 = rest_d - rdh;
    auto W3 = rest_h - rdh;
    auto W4 = rdh;    
    
   
    auto df_P1 = d_floor(scaled_P1, sigma, half_range_bot_d, half_range_top_d, d_sigmoid()) - ones;
    auto df_P2 = d_floor(scaled_P2, sigma, half_range_bot_h, half_range_top_h, d_sigmoid()) - ones;
    
    auto W1_Pd = df_P1 * (weight - rhW);
    auto W2_Pd = -W1_Pd;
    auto W3_Pd = df_P1 * rhW;
    auto W4_Pd = -W3_Pd;
    
    auto W1_Ph = df_P2 * (weight - rdW) ;
    auto W2_Ph = df_P2 * rdW;
    auto W3_Ph = -W1_Ph;
    auto W4_Ph = -W2_Ph;
    

    const int num_kernels =  channels_out * channels_in * kernel_d * kernel_h * kernel_w;    
    AT_DISPATCH_FLOATING_TYPES(weight.type(), "dcls_construct_3_2d_backward_cuda", [&] {
             
        interpolation_grad_kernel<scalar_t><<<GET_BLOCKS(num_kernels), 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                     num_kernels,
                                     grad_output.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     P_d.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     P_h.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W1.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W2.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W3.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W4.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     channels_out, channels_in,
                                     kernel_d, kernel_h, kernel_w,
                                     depth_out, height_out,                                
                                     grad_weight.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>());
        interpolation_grad_kernel<scalar_t><<<GET_BLOCKS(num_kernels), 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                     num_kernels,
                                     grad_output.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     P_d.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     P_h.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W1_Pd.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W2_Pd.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W3_Pd.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W4_Pd.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     channels_out, channels_in,
                                     kernel_d, kernel_h, kernel_w,
                                     depth_out, height_out,                                   
                                     grad_P1.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>());                          
        interpolation_grad_kernel<scalar_t><<<GET_BLOCKS(num_kernels), 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                     num_kernels,
                                     grad_output.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     P_d.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     P_h.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W1_Ph.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W2_Ph.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W3_Ph.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W4_Ph.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     channels_out, channels_in,
                                     kernel_d, kernel_h, kernel_w,
                                     depth_out, height_out,                                   
                                     grad_P2.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>());                                     
        
    });


    return {grad_weight,
            grad_P1*scaling_d,
            grad_P2*scaling_h};
}
