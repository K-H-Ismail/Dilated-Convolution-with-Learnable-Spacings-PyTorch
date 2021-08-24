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
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> P_h,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> P_w,    
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> W1, 
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> W2,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> W3, 
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> W4,    
    const int ch_out, const int ch_in,
    const int kernel_h, const int kernel_w,
    const int height_out, const int width_out,    
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits>  interpolated_weight) {
  CUDA_KERNEL_LOOP(index, n) {
    int w_out = index % kernel_w;
    int h_out = (index / kernel_w) % kernel_h;
    int channel_in = (index / kernel_h / kernel_w) % ch_in;
    int channel_out = (index / kernel_h / kernel_w / ch_in) % ch_out;
      
    int p_h = P_h[channel_out][channel_in][h_out][w_out];
    int p_w = P_w[channel_out][channel_in][h_out][w_out];
    int p_h_next = p_h + 1;
    int p_w_next = p_w + 1;      
       
    if(p_h >= 0 & p_h < height_out & p_w >= 0 & p_w < width_out)
    {   
        interpolated_weight[channel_out][channel_in][p_h][p_w] +=  W1[channel_out][channel_in][h_out][w_out];
        if(p_h_next < height_out) 
            interpolated_weight[channel_out][channel_in][p_h_next][p_w] += W2[channel_out][channel_in][h_out][w_out];
        if(p_w_next < width_out) 
            interpolated_weight[channel_out][channel_in][p_h][p_w_next] += W3[channel_out][channel_in][h_out][w_out];
        if(p_h_next < height_out & p_w_next < width_out) 
            interpolated_weight[channel_out][channel_in][p_h_next][p_w_next] += W4[channel_out][channel_in][h_out][w_out];
    }
      
  }
}

template <typename scalar_t>
__global__ void interpolation_grad_kernel(
    const int n,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_output,    
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> P_h,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> P_w,  
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> W1, 
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> W2,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> W3, 
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> W4,    
    const int ch_out, const int ch_in,
    const int kernel_h, const int kernel_w,
    const int height_out, const int width_out,     
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits>  interpolated_weight) {
  CUDA_KERNEL_LOOP(index, n) {
    int w_out = index % kernel_w;
    int h_out = (index / kernel_w) % kernel_h;
    int channel_in = (index / kernel_h / kernel_w) % ch_in;
    int channel_out = (index / kernel_h / kernel_w / ch_in) % ch_out;
      
    int p_h = P_h[channel_out][channel_in][h_out][w_out];
    int p_w = P_w[channel_out][channel_in][h_out][w_out];
    int p_h_next = p_h + 1;
    int p_w_next = p_w + 1;       

    if(p_h >= 0 & p_h < height_out & p_w >= 0 & p_w < width_out)
    {     
        interpolated_weight[channel_out][channel_in][h_out][w_out] += 
            grad_output[channel_out][channel_in][p_h][p_w] * W1[channel_out][channel_in][h_out][w_out];

        if(p_h_next < height_out)
            interpolated_weight[channel_out][channel_in][h_out][w_out] +=        
            grad_output[channel_out][channel_in][p_h_next][p_w] * W2[channel_out][channel_in][h_out][w_out];

        if(p_w_next < width_out)
            interpolated_weight[channel_out][channel_in][h_out][w_out] +=        
            grad_output[channel_out][channel_in][p_h][p_w_next] * W3[channel_out][channel_in][h_out][w_out];

        if(p_h_next < height_out & p_w_next < width_out)
            interpolated_weight[channel_out][channel_in][h_out][w_out] += 
            grad_output[channel_out][channel_in][p_h_next][p_w_next] * W4[channel_out][channel_in][h_out][w_out];
    }
      
  }
}

torch::Tensor  dcls_2d_cuda_forward(  
    torch::Tensor weight,
    torch::Tensor P1,
    torch::Tensor P2,
    const int dilation_h, const int dilation_w
    ) {
    
    const int channels_out = weight.size(0);
    const int channels_in = weight.size(1);    
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);
    
    // Suitable for Kaiming uniform initialization
    auto scaling_h = sqrt(kernel_h * kernel_w * channels_in * dilation_h * dilation_h)/2;
    auto scaling_w = sqrt(kernel_h * kernel_w * channels_in * dilation_w * dilation_w)/2;    
 
    const int half_range_bot_h = (dilation_h*kernel_h)/2;

    const int half_range_bot_w = (dilation_w*kernel_w)/2;
    
    auto scaled_P1 = P1*scaling_h + at::arange(-half_range_bot_h /*+ dilation_h/4*/,half_range_bot_h /*+ 1e-7*/,dilation_h, weight.options())
                            .repeat({kernel_w,1})
                            .t()
                            .repeat({channels_out,channels_in,1,1});
    auto scaled_P2 = P2*scaling_w + at::arange(-half_range_bot_w /*+ dilation_w/4*/,half_range_bot_w /*+ 1e-7*/,dilation_w, weight.options())
                            .repeat({kernel_h,1})
                            .repeat({channels_out,channels_in,1,1});
        
    const int height_out = dilation_h * (kernel_h-1) + 1;
    const int width_out = dilation_w * (kernel_w-1) + 1;
    
    auto P_h = scaled_P1.floor();
    auto P_w = scaled_P2.floor();    
    
    P_h += (dilation_h*kernel_h)/2 ;
    P_w += (dilation_w*kernel_w)/2 ;
    
    P_h = P_h.clamp(0,height_out-1); 
    P_w = P_w.clamp(0,width_out-1);    
    
    auto rest_h = (scaled_P1 + (dilation_h*kernel_h)/2).clamp(0,height_out-1) - P_h; 
    auto rest_w = (scaled_P2 + (dilation_w*kernel_w)/2).clamp(0,width_out-1) - P_w;    
    
    auto rhW = rest_h * weight;
    auto rwW = rest_w * weight;
    auto rhwW = rest_h * rwW;    
    auto W1 = weight - rhW - rwW + rhwW;
    auto W2 = rhW - rhwW;
    auto W3 = rwW - rhwW;
    auto W4 = rhwW;    
    
    auto output = torch::zeros({channels_out, channels_in, height_out, width_out}, weight.options());
    
    const int num_kernels =  channels_out * channels_in * kernel_h * kernel_w ;
    AT_DISPATCH_FLOATING_TYPES(weight.type(), "dcls_2d_forward_cuda", [&] {
          
        interpolation_kernel<scalar_t><<<GET_BLOCKS(num_kernels), 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                     num_kernels,
                                     P_h.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                     P_w.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                     W1.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                     W2.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                     W3.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                     W4.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                     channels_out, channels_in,
                                     kernel_h, kernel_w,
                                     height_out, width_out,
                                     output.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
    });    
    return output;
}

std::vector<torch::Tensor> dcls_2d_cuda_backward(   
    torch::Tensor weight,
    torch::Tensor P1,
    torch::Tensor P2,
    torch::Tensor grad_output,      
    const int dilation_h, const int dilation_w
    ) {
    
    auto grad_weight = torch::zeros_like(weight);
    auto grad_P1 = torch::zeros_like(P1);
    auto grad_P2 = torch::zeros_like(P2);     
        
    const int channels_out = weight.size(0);
    const int channels_in = weight.size(1);    
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);
    
    const int half_range_bot_h = (dilation_h*kernel_h)/2;
    const int half_range_top_h = half_range_bot_h - (dilation_h*kernel_h+1)%2;    

    const int half_range_bot_w = (dilation_w*kernel_w)/2;
    const int half_range_top_w = half_range_bot_w - (dilation_w*kernel_w+1)%2;
    
    // Suitable for Kaiming uniform initialization
    auto scaling_h = sqrt(kernel_h * kernel_w * channels_in * dilation_h * dilation_h)/2;
    auto scaling_w = sqrt(kernel_h * kernel_w * channels_in * dilation_w * dilation_w)/2;  
    
    auto scaled_P1 = P1*scaling_h + at::arange(-half_range_bot_h /*+ dilation_h/4*/,half_range_bot_h /*+ 1e-7*/,dilation_h, weight.options())
                            .repeat({kernel_w,1})
                            .t()
                            .repeat({channels_out,channels_in,1,1});
    auto scaled_P2 = P2*scaling_w + at::arange(-half_range_bot_w /*+ dilation_w/4*/,half_range_bot_w /*+ 1e-7*/,dilation_w, weight.options())
                            .repeat({kernel_h,1})
                            .repeat({channels_out,channels_in,1,1});
        
    
    const int height_out = dilation_h * (kernel_h-1) + 1;
    const int width_out = dilation_w * (kernel_w-1) + 1;
    
    auto P_h = scaled_P1.floor();
    auto P_w = scaled_P2.floor();    
    
    P_h += (dilation_h*kernel_h)/2 ;
    P_w += (dilation_w*kernel_w)/2 ;
    
    P_h = P_h.clamp(0,height_out-1); 
    P_w = P_w.clamp(0,width_out-1);    
    
    auto rest_h = scaled_P1 + (dilation_h*kernel_h)/2;
    auto mask_h = rest_h.ge(0) * rest_h.le(height_out-1);
    rest_h = rest_h.clamp(0,height_out-1) - P_h; 
    auto rest_w = scaled_P2 + (dilation_w*kernel_w)/2;
    auto mask_w = rest_w.ge(0) * rest_w.le(width_out-1);
    rest_w = rest_w.clamp(0,width_out-1) - P_w;    
    


    auto rhW = rest_h * weight * mask_w;
    auto rwW = rest_w * weight * mask_h;
    auto rhw = rest_h * rest_w;
    auto rhwW = rhw * weight;    
    
    auto ones = at::ones_like(rest_h, weight.options());
        
    auto W1 = ones - rest_h - rest_w + rhw;
    auto W2 = rest_h - rhw;
    auto W3 = rest_w - rhw;
    auto W4 = rhw; 
    
    auto W1_Ph = -weight * mask_h + rwW;
    auto W2_Ph = -W1_Ph;
    auto W3_Ph = -rwW;
    auto W4_Ph = -W3_Ph;
    
    auto W1_Pw = -weight * mask_w + rhW ;
    auto W2_Pw = -rhW;
    auto W3_Pw = -W1_Pw;
    auto W4_Pw = -W2_Pw;
    
    
    const int num_kernels = channels_out * channels_in * kernel_h * kernel_w;    
    AT_DISPATCH_FLOATING_TYPES(weight.type(), "dcls_2d_backward_cuda", [&] {
             
        interpolation_grad_kernel<scalar_t><<<GET_BLOCKS(num_kernels), 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                     num_kernels,
                                     grad_output.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                     P_h.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                     P_w.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                     W1.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                     W2.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                     W3.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                     W4.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                     channels_out, channels_in,
                                     kernel_h, kernel_w, 
                                     height_out, width_out,                                 
                                     grad_weight.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
        interpolation_grad_kernel<scalar_t><<<GET_BLOCKS(num_kernels), 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                     num_kernels,
                                     grad_output.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                     P_h.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                     P_w.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                     W1_Ph.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                     W2_Ph.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                     W3_Ph.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                     W4_Ph.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                     channels_out, channels_in,
                                     kernel_h, kernel_w, 
                                     height_out, width_out,                                 
                                     grad_P1.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
        interpolation_grad_kernel<scalar_t><<<GET_BLOCKS(num_kernels), 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                     num_kernels,
                                     grad_output.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                     P_h.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                     P_w.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                     W1_Pw.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                     W2_Pw.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                     W3_Pw.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                     W4_Pw.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                     channels_out, channels_in,
                                     kernel_h, kernel_w, 
                                     height_out, width_out,                                 
                                     grad_P2.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>()); 
        
    });


    return {grad_weight,
            grad_P1*scaling_h,
            grad_P2*scaling_w};
}
