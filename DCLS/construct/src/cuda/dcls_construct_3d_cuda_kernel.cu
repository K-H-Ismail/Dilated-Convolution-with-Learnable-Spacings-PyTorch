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
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> P_w,    
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> W1, 
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> W2,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> W3, 
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> W4,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> W5, 
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> W6,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> W7, 
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> W8,    
    const int ch_out, const int ch_in,
    const int kernel_d, const int kernel_h, const int kernel_w,    
    const int depth_out, const int height_out, const int width_out,    
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> interpolated_weight) {
  CUDA_KERNEL_LOOP(index, n) {
    int w_out = index % kernel_w;
    int h_out = (index / kernel_w) % kernel_h;
    int d_out = ((index / kernel_w) / kernel_h) % kernel_d;      
    int channel_in = (index / kernel_d / kernel_h / kernel_w) % ch_in;
    int channel_out = (index / kernel_d / kernel_h / kernel_w / ch_in) % ch_out;

    int p_d = P_d[channel_out][channel_in][d_out][h_out][w_out];      
    int p_h = P_h[channel_out][channel_in][d_out][h_out][w_out];
    int p_w = P_w[channel_out][channel_in][d_out][h_out][w_out];
    int p_d_next = p_d + 1;      
    int p_h_next = p_h + 1;
    int p_w_next = p_w + 1; 
   
      
    if(p_d >= 0 & p_d < depth_out & p_h >= 0 & p_h < height_out & p_w >= 0 & p_w < width_out)
    {
        interpolated_weight[channel_out][channel_in][p_d][p_h][p_w] +=  W1[channel_out][channel_in][d_out][h_out][w_out];
        if(p_d_next < depth_out) 
            interpolated_weight[channel_out][channel_in][p_d_next][p_h][p_w] += W2[channel_out][channel_in][d_out][h_out][w_out];
        if(p_h_next < height_out) 
            interpolated_weight[channel_out][channel_in][p_d][p_h_next][p_w] += W3[channel_out][channel_in][d_out][h_out][w_out];
        if(p_d_next < depth_out & p_h_next < height_out) 
            interpolated_weight[channel_out][channel_in][p_d_next][p_h_next][p_w] += W4[channel_out][channel_in][d_out][h_out][w_out];
        if(p_w_next < width_out)
            interpolated_weight[channel_out][channel_in][p_d][p_h][p_w_next] +=  W5[channel_out][channel_in][d_out][h_out][w_out];
        if(p_d_next < depth_out & p_w_next < width_out) 
            interpolated_weight[channel_out][channel_in][p_d_next][p_h][p_w_next] += W6[channel_out][channel_in][d_out][h_out][w_out];
        if(p_h_next < height_out & p_w_next < width_out) 
            interpolated_weight[channel_out][channel_in][p_d][p_h_next][p_w_next] += W7[channel_out][channel_in][d_out][h_out][w_out];
        if(p_d_next < depth_out & p_h_next < height_out & p_w_next < width_out) 
            interpolated_weight[channel_out][channel_in][p_d_next][p_h_next][p_w_next] += W8[channel_out][channel_in][d_out][h_out][w_out];        
    }
  }
}

template <typename scalar_t>
__global__ void interpolation_grad_kernel(
    const int n,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_output,    
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> P_d,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> P_h,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> P_w,    
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> W1, 
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> W2,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> W3, 
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> W4,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> W5, 
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> W6,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> W7, 
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> W8,    
    const int ch_out, const int ch_in,
    const int kernel_d, const int kernel_h, const int kernel_w,    
    const int depth_out, const int height_out, const int width_out,  
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits>  interpolated_weight) {
  CUDA_KERNEL_LOOP(index, n) {
    int w_out = index % kernel_w;
    int h_out = (index / kernel_w) % kernel_h;
    int d_out = ((index / kernel_w) / kernel_h) % kernel_d;      
    int channel_in = (index / kernel_d / kernel_h / kernel_w) % ch_in;
    int channel_out = (index / kernel_d / kernel_h / kernel_w / ch_in) % ch_out;
      
    int p_d = P_d[channel_out][channel_in][d_out][h_out][w_out];      
    int p_h = P_h[channel_out][channel_in][d_out][h_out][w_out];
    int p_w = P_w[channel_out][channel_in][d_out][h_out][w_out];
    int p_d_next = p_d + 1;      
    int p_h_next = p_h + 1;
    int p_w_next = p_w + 1; 
   
      
    if(p_d >= 0 & p_d < depth_out & p_h >= 0 & p_h < height_out & p_w >= 0 & p_w < width_out)
    {
        interpolated_weight[channel_out][channel_in][d_out][h_out][w_out] +=
            grad_output[channel_out][channel_in][p_d][p_h][p_w] *  W1[channel_out][channel_in][d_out][h_out][w_out];
        if(p_d_next < depth_out)
            interpolated_weight[channel_out][channel_in][d_out][h_out][w_out] +=
                grad_output[channel_out][channel_in][p_d_next][p_h][p_w] * W2[channel_out][channel_in][d_out][h_out][w_out];
        if(p_h_next < height_out)
            interpolated_weight[channel_out][channel_in][d_out][h_out][w_out] +=
                grad_output[channel_out][channel_in][p_d][p_h_next][p_w] * W3[channel_out][channel_in][d_out][h_out][w_out];
        if(p_d_next < depth_out & p_h_next < height_out)
            interpolated_weight[channel_out][channel_in][d_out][h_out][w_out] +=
                grad_output[channel_out][channel_in][p_d_next][p_h_next][p_w] * W4[channel_out][channel_in][d_out][h_out][w_out];
        if(p_w_next < width_out)
            interpolated_weight[channel_out][channel_in][d_out][h_out][w_out] +=
                grad_output[channel_out][channel_in][p_d][p_h][p_w_next] *  W5[channel_out][channel_in][d_out][h_out][w_out];
        if(p_d_next < depth_out & p_w_next < width_out)
            interpolated_weight[channel_out][channel_in][d_out][h_out][w_out] +=
                grad_output[channel_out][channel_in][p_d_next][p_h][p_w_next] * W6[channel_out][channel_in][d_out][h_out][w_out];
        if(p_h_next < height_out & p_w_next < width_out)
            interpolated_weight[channel_out][channel_in][d_out][h_out][w_out] +=
                grad_output[channel_out][channel_in][p_d][p_h_next][p_w_next] * W7[channel_out][channel_in][d_out][h_out][w_out];
        if(p_d_next < depth_out & p_h_next < height_out & p_w_next < width_out)
            interpolated_weight[channel_out][channel_in][d_out][h_out][w_out] +=
                grad_output[channel_out][channel_in][p_d_next][p_h_next][p_w_next] * W8[channel_out][channel_in][d_out][h_out][w_out];        
    }    
  }
}

torch::Tensor  dcls_construct_3d_cuda_forward(  
    torch::Tensor weight,
    torch::Tensor P1,
    torch::Tensor P2,
    torch::Tensor P3,    
    const int dilation_d,  
    const int dilation_h,  
    const int dilation_w    
    ) {
    
    const int channels_out = weight.size(0);
    const int channels_in = weight.size(1);    
    const int kernel_d = weight.size(2);
    const int kernel_h = weight.size(3);
    const int kernel_w = weight.size(4);    
 
    const int half_range_bot_d = dilation_d*kernel_d/2;
    
    const int half_range_bot_h = dilation_h*kernel_h/2;

    const int half_range_bot_w = dilation_w*kernel_w/2;
    
    auto scaling_d = sqrt(kernel_h * kernel_w * kernel_d * channels_in * dilation_d * dilation_d / 4);
    auto scaling_h = sqrt(kernel_h * kernel_w * kernel_d * channels_in * dilation_h * dilation_h / 4); 
    auto scaling_w = sqrt(kernel_h * kernel_w * kernel_d * channels_in * dilation_w * dilation_w / 4);     

    auto scaled_P1 = P1*scaling_d + at::arange(-half_range_bot_d + dilation_d/2,half_range_bot_d + 1e-7,dilation_d, weight.options())
                            .repeat({kernel_h,kernel_w,1})
                            .permute({2,0,1})
                            .repeat({channels_out,channels_in,1,1,1});
    auto scaled_P2 = P2*scaling_h + at::arange(-half_range_bot_h + dilation_h/2,half_range_bot_h + 1e-7,dilation_h, weight.options())
                            .repeat({kernel_d,kernel_w,1})
                            .permute({0,2,1})
                            .repeat({channels_out,channels_in,1,1,1});
    auto scaled_P3 = P3*scaling_w + at::arange(-half_range_bot_w + dilation_w/2,half_range_bot_w + 1e-7,dilation_w, weight.options())
                            .repeat({kernel_d,kernel_h,1})
                            .permute({0,1,2})
                            .repeat({channels_out,channels_in,1,1,1});    
                            
    auto P_d = scaled_P1.floor();
    auto rest_d = scaled_P1 - P_d;
    
    auto P_h = scaled_P2.floor();
    auto rest_h = scaled_P2 - P_h;
    
    auto P_w = scaled_P3.floor();
    auto rest_w = scaled_P3 - P_w;

    const int depth_out = dilation_d * kernel_d + (dilation_d+1)%2;
    const int height_out = dilation_h * kernel_h + (dilation_h+1)%2;
    const int width_out = dilation_w * kernel_w + (dilation_w+1)%2;
    
    P_d += dilation_d*kernel_d/2 ;    
    P_h += dilation_h*kernel_h/2 ;
    P_w += dilation_w*kernel_w/2 ;
    
    P_d = P_d.clamp(0,depth_out-1); 
    P_h = P_h.clamp(0,height_out-1);
    P_w = P_w.clamp(0,width_out-1);    

    auto rdW = rest_d * weight;    
    auto rhW = rest_h * weight;
    auto rwW = rest_w * weight;
    auto rdhW = rest_d * rhW;    
    auto rdwW = rest_d * rwW;
    auto rhwW = rest_h * rwW;
    auto rdhwW = rest_d * rhwW;
    
    auto W1 = weight -rdW - rhW - rwW + rdhW + rdwW + rhwW - rdhwW;
    auto W2 = rdW - rdwW - rdhW + rdhwW;
    auto W3 = rhW - rdhW - rhwW + rdhwW;
    auto W4 = rdhW - rdhwW;
    auto W5 = rwW - rhwW - rdwW + rdhwW;
    auto W6 = rdwW - rdhwW;
    auto W7 = rhwW - rdhwW;
    auto W8 = rdhwW;     
    
   
    auto output = torch::zeros({channels_out, channels_in, depth_out, height_out, width_out}, weight.options());
    
    const int num_kernels =  channels_out * channels_in * kernel_d * kernel_h * kernel_w;
    AT_DISPATCH_FLOATING_TYPES(weight.type(), "dcls_construct_3d_forward_cuda", [&] {
          
        interpolation_kernel<scalar_t><<<GET_BLOCKS(num_kernels), 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                     num_kernels,
                                     P_d.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),            
                                     P_h.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     P_w.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W1.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W2.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W3.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W4.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W5.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W6.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W7.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W8.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),            
                                     channels_out, channels_in,
                                     kernel_d, kernel_h, kernel_w,
                                     depth_out, height_out, width_out,
                                     output.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>());
    });    
    return output;
}

std::vector<torch::Tensor> dcls_construct_3d_cuda_backward(   
    torch::Tensor weight,
    torch::Tensor P1,
    torch::Tensor P2,
    torch::Tensor P3,    
    torch::Tensor grad_output,
    const int dilation_d,  
    const int dilation_h,  
    const int dilation_w    
    ) {
    
    auto grad_weight = torch::zeros_like(weight);
    auto grad_P1 = torch::zeros_like(P1);
    auto grad_P2 = torch::zeros_like(P2);
    auto grad_P3 = torch::zeros_like(P3);    
        
    const int channels_out = weight.size(0);
    const int channels_in = weight.size(1);    
    const int kernel_d = weight.size(2);
    const int kernel_h = weight.size(3);
    const int kernel_w = weight.size(4);
    
    const int half_range_bot_d = dilation_d*kernel_d/2;
    const int half_range_top_d = half_range_bot_d - (dilation_d*kernel_d + 1)%2;    
    
    const int half_range_bot_h = dilation_h*kernel_h/2;
    const int half_range_top_h = half_range_bot_h - (dilation_h*kernel_h + 1)%2;    

    const int half_range_bot_w = dilation_w*kernel_w/2;
    const int half_range_top_w = half_range_bot_w - (dilation_w*kernel_w + 1)%2;  
    
    auto scaling_d = sqrt(kernel_h * kernel_w * kernel_d * channels_in * dilation_d * dilation_d / 4);
    auto scaling_h = sqrt(kernel_h * kernel_w * kernel_d * channels_in * dilation_h * dilation_h / 4); 
    auto scaling_w = sqrt(kernel_h * kernel_w * kernel_d * channels_in * dilation_w * dilation_w / 4);    
    
    
    auto scaled_P1 = P1*scaling_d + at::arange(-half_range_bot_d + dilation_d/2,half_range_bot_d + 1e-7,dilation_d, weight.options())
                            .repeat({kernel_h,kernel_w,1})
                            .permute({2,0,1})
                            .repeat({channels_out,channels_in,1,1,1});
    auto scaled_P2 = P2*scaling_h + at::arange(-half_range_bot_h + dilation_h/2,half_range_bot_h + 1e-7,dilation_h, weight.options())
                            .repeat({kernel_d,kernel_w,1})
                            .permute({0,2,1})
                            .repeat({channels_out,channels_in,1,1,1});
    auto scaled_P3 = P3*scaling_w + at::arange(-half_range_bot_w + dilation_w/2,half_range_bot_w + 1e-7,dilation_w, weight.options())
                            .repeat({kernel_d,kernel_h,1})
                            .permute({0,1,2})
                            .repeat({channels_out,channels_in,1,1,1});     
                            
    auto P_d = scaled_P1.floor();
    auto rest_d = scaled_P1 - P_d;
    
    auto P_h = scaled_P2.floor();
    auto rest_h = scaled_P2 - P_h;
    
    auto P_w = scaled_P3.floor();
    auto rest_w = scaled_P3 - P_w;

    const int depth_out = dilation_d * kernel_d + (dilation_d+1)%2;
    const int height_out = dilation_h * kernel_h + (dilation_h+1)%2;
    const int width_out = dilation_w * kernel_w + (dilation_w+1)%2;
    
    P_d += dilation_d*kernel_d/2 ;    
    P_h += dilation_h*kernel_h/2 ;
    P_w += dilation_w*kernel_w/2 ;
    
    P_d = P_d.clamp(0,depth_out-1); 
    P_h = P_h.clamp(0,height_out-1);
    P_w = P_w.clamp(0,width_out-1);    
   
    
    auto rdW = rest_d * weight;    
    auto rhW = rest_h * weight;
    auto rwW = rest_w * weight;
    auto rdh = rest_d * rest_h;    
    auto rdhW = rest_d * rhW;
    auto rdw = rest_d * rest_w;     
    auto rdwW = rest_d * rwW;
    auto rhw = rest_h * rest_w;     
    auto rhwW = rest_h * rwW;
    auto rdhw = rdh * rest_w;     
    auto rdhwW = rest_d * rhwW;

    auto ones = at::ones_like(rest_d, weight.options());
    auto sigma = 0.5*ones; 
    
    auto W1 = ones -rest_d - rest_h - rest_w + rdh + rdw + rhw - rdhw;
    auto W2 = rest_d - rdw - rdh + rdhw;
    auto W3 = rest_h - rdh - rhw + rdhw;
    auto W4 = rdh - rdhw;
    auto W5 = rest_w - rhw - rdw + rdhw;
    auto W6 = rdw - rdhw;
    auto W7 = rhw - rdhw;
    auto W8 = rdhw; 
    
   
    auto df_P1 = d_floor(scaled_P1, sigma, half_range_bot_d, half_range_top_d, d_sigmoid()) - ones;
    auto df_P2 = d_floor(scaled_P2, sigma, half_range_bot_h, half_range_top_h, d_sigmoid()) - ones;
    auto df_P3 = d_floor(scaled_P2, sigma, half_range_bot_w, half_range_top_w, d_sigmoid()) - ones;    
    
    auto W1_Pd = df_P1 * (weight - rhW - rwW + rhwW);
    auto W2_Pd = -W1_Pd;
    auto W3_Pd = df_P1 * (rhW - rhwW);
    auto W4_Pd = -W3_Pd;
    auto W5_Pd = df_P1 * (rwW - rhwW);
    auto W6_Pd = -W5_Pd;
    auto W7_Pd = df_P1 * rhwW ;
    auto W8_Pd = -W7_Pd;    
    
    auto W1_Ph = df_P2 * (weight - rdW - rwW + rdwW);
    auto W2_Ph = df_P2 * (rdW - rdwW);
    auto W3_Ph = -W1_Ph;
    auto W4_Ph = -W2_Ph;
    auto W5_Ph = df_P2 * (rwW - rdwW);
    auto W6_Ph = df_P2 * rdwW;
    auto W7_Ph = -W5_Ph;
    auto W8_Ph = -W6_Ph;
    
    auto W1_Pw = df_P3 * (weight - rdW - rhW + rdhW);
    auto W2_Pw = df_P3 * (rdW - rdhW);
    auto W3_Pw = df_P3 * (rhW - rdhW);
    auto W4_Pw = df_P3 * rdhW;
    auto W5_Pw = -W1_Pw;
    auto W6_Pw = -W2_Pw;
    auto W7_Pw = -W3_Pw;
    auto W8_Pw = -W4_Pw;     
    

    const int num_kernels =  channels_out * channels_in * kernel_d * kernel_h * kernel_w;    
    AT_DISPATCH_FLOATING_TYPES(weight.type(), "dcls_construct_3d_backward_cuda", [&] {
             
        interpolation_grad_kernel<scalar_t><<<GET_BLOCKS(num_kernels), 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                     num_kernels,
                                     grad_output.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     P_d.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),            
                                     P_h.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     P_w.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W1.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W2.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W3.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W4.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W5.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W6.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W7.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W8.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),            
                                     channels_out, channels_in,
                                     kernel_d, kernel_h, kernel_w,
                                     depth_out, height_out, width_out,                          
                                     grad_weight.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>());

        interpolation_grad_kernel<scalar_t><<<GET_BLOCKS(num_kernels), 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                     num_kernels,
                                     grad_output.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     P_d.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),            
                                     P_h.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     P_w.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W1_Pd.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W2_Pd.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W3_Pd.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W4_Pd.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W5_Pd.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W6_Pd.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W7_Pd.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W8_Pd.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),            
                                     channels_out, channels_in,
                                     kernel_d, kernel_h, kernel_w,
                                     depth_out, height_out, width_out,                                    
                                     grad_P1.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>());
                                     
        interpolation_grad_kernel<scalar_t><<<GET_BLOCKS(num_kernels), 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                     num_kernels,
                                     grad_output.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     P_d.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),            
                                     P_h.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     P_w.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W1_Ph.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W2_Ph.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W3_Ph.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W4_Ph.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W5_Ph.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W6_Ph.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W7_Ph.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W8_Ph.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),            
                                     channels_out, channels_in,
                                     kernel_d, kernel_h, kernel_w,
                                     depth_out, height_out, width_out,                                    
                                     grad_P2.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>()); 
                                     
        interpolation_grad_kernel<scalar_t><<<GET_BLOCKS(num_kernels), 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                     num_kernels,
                                     grad_output.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     P_d.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),            
                                     P_h.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     P_w.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W1_Pw.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W2_Pw.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W3_Pw.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W4_Pw.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W5_Pw.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W6_Pw.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W7_Pw.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                     W8_Pw.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),            
                                     channels_out, channels_in,
                                     kernel_d, kernel_h, kernel_w,
                                     depth_out, height_out, width_out,                                    
                                     grad_P3.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>());                                     
        
    });


    return {grad_weight,
            grad_P1*scaling_d,
            grad_P2*scaling_h,
            grad_P3*scaling_w};
}
