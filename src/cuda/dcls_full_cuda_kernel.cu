#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <math.h>
#include <vector>


#define CUDA_KERNEL_LOOP(i, n)                                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);                 \
       i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
const int kMaxGridNum = 65535;
inline int GET_BLOCKS(const int N) {
  return std::min(kMaxGridNum, (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);
}

torch::Tensor d_sigmoid(torch::Tensor z, torch::Tensor sigma) {
  auto s = torch::sigmoid(sigma * z);
  return (1.0 - s) * s * sigma;
}


torch::Tensor d_ceil(torch::Tensor z, torch::Tensor sigma, const int bot, const int top) {
  auto s = torch::zeros_like(z);
  for (int i = 1-bot; i < top; i++) 
  { 
      s += d_sigmoid(z + static_cast<double>(i), sigma);
  }
  return s;
}

torch::Tensor d_floor(torch::Tensor z, torch::Tensor sigma, const int bot, const int top) {
  auto s = torch::zeros_like(z);
  for (int i = -bot; i < top-1; i++) 
  { 
      s += d_sigmoid(z + static_cast<double>(i), sigma);
  }
  return s;
}


template <typename scalar_t>
__global__ void interpolation_kernel(
    const int n,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> weight,
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
      
    scalar_t w = weight[channel_out][channel_in][h_out][w_out];
    int p_h = P_h[channel_out][channel_in][h_out][w_out];
    int p_w = P_w[channel_out][channel_in][h_out][w_out];
    int p_h_next = p_h + 1;
    int p_w_next = p_w + 1;      
   
    interpolated_weight[channel_out][channel_in][p_h][p_w] += w * W1[channel_out][channel_in][h_out][w_out];
    if(p_h_next < height_out) 
        interpolated_weight[channel_out][channel_in][p_h_next][p_w] += w * W2[channel_out][channel_in][h_out][w_out];
    if(p_w_next < width_out) 
        interpolated_weight[channel_out][channel_in][p_h][p_w_next] += w * W3[channel_out][channel_in][h_out][w_out];
    if(p_h_next < height_out & p_w_next < width_out) 
        interpolated_weight[channel_out][channel_in][p_h_next][p_w_next] += w * W4[channel_out][channel_in][h_out][w_out];    
      
  }
}

template <typename scalar_t>
__global__ void interpolation_grad_kernel(
    const int n,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_output,    
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> weight,
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
      
    scalar_t w = weight[channel_out][channel_in][h_out][w_out];
    int p_h = P_h[channel_out][channel_in][h_out][w_out];
    int p_w = P_w[channel_out][channel_in][h_out][w_out];
    int p_h_next = p_h + 1;
    int p_w_next = p_w + 1;       
      
    scalar_t g1 = grad_output[channel_out][channel_in][p_h][p_w]; 
    scalar_t g2 = (p_h_next < height_out) ? grad_output[channel_out][channel_in][p_h_next][p_w] : 0; 
    scalar_t g3 = (p_w_next < width_out) ? grad_output[channel_out][channel_in][p_h][p_w_next] : 0; 
    scalar_t g4 = (p_h_next < height_out & p_w_next < width_out) ? grad_output[channel_out][channel_in][p_h_next][p_w_next] : 0;       
    
    interpolated_weight[channel_out][channel_in][h_out][w_out]+= g1 * W1[channel_out][channel_in][h_out][w_out]
                                                                 + g2 * W2[channel_out][channel_in][h_out][w_out]
                                                                 + g3 * W3[channel_out][channel_in][h_out][w_out]
                                                                 + g4 * W4[channel_out][channel_in][h_out][w_out]; 
      
  }
}

torch::Tensor  dcls_full_cuda_forward(  
    torch::Tensor weight,
    torch::Tensor P1,
    torch::Tensor P2,
    const int dilation_h, const int dilation_w
    ) {
    
    const int channels_out = weight.size(0);
    const int channels_in = weight.size(1);    
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);
 

    const int half_range_bot_h = dilation_h*kernel_h/2;
    const int half_range_top_h = half_range_bot_h - (dilation_h*kernel_h + 1)%2;

    const int half_range_bot_w = dilation_w*kernel_w/2;
    const int half_range_top_w = half_range_bot_w - (dilation_w*kernel_w +1)%2;
    
    auto P_h = at::clamp(at::floor(P1),-half_range_bot_h,half_range_top_h);
    auto rest_h = at::clamp(P1,-half_range_bot_h,half_range_top_h) - P_h;
        
    auto P_w = at::clamp(at::floor(P2),-half_range_bot_w,half_range_top_w);
    auto rest_w = at::clamp(P2,-half_range_bot_w,half_range_top_w) - P_w;
    
    P_h += dilation_h*kernel_h/2 ;
    P_w += dilation_w*kernel_w/2 ;
    
    auto ones = at::ones_like(rest_h, weight.options());    
    auto W1 = (ones - rest_h) * (ones - rest_w);
    auto W2 = rest_h * (ones - rest_w);
    auto W3 = (ones - rest_h) * rest_w;
    auto W4 = rest_h * rest_w;    
    
    const int height_out = dilation_h * (kernel_h - 1) + 1;
    const int width_out = dilation_w * (kernel_w - 1) + 1;
   
    auto output = torch::zeros({channels_out, channels_in, height_out, width_out}, weight.options());
    
    const int num_kernels =  channels_out * channels_in * kernel_h * kernel_w ;
    AT_DISPATCH_FLOATING_TYPES(weight.type(), "dcls_full_forward_cuda", [&] {
          
        interpolation_kernel<scalar_t><<<GET_BLOCKS(num_kernels), 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                     num_kernels,
                                     weight.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
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

std::vector<torch::Tensor> dcls_full_cuda_backward(   
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
    
    const int half_range_bot_h = dilation_h*kernel_h/2;
    const int half_range_top_h = half_range_bot_h - (dilation_h*kernel_h+1)%2;

    const int half_range_bot_w = dilation_w*kernel_w/2;
    const int half_range_top_w = half_range_bot_w - (dilation_w*kernel_w+1)%2;
    
    auto P_h = at::clamp(at::floor(P1),-half_range_bot_h,half_range_top_h);
    auto rest_h = -P_h + at::clamp(P1,-half_range_bot_h,half_range_top_h);
        
    auto P_w = at::clamp(at::floor(P2),-half_range_bot_w,half_range_top_w);
    auto rest_w = -P_w + at::clamp(P2,-half_range_bot_w,half_range_top_w);
    
    P_h += dilation_h*kernel_h/2;
    P_w += dilation_w*kernel_w/2;    
    
    auto sigma = 0.5*at::ones_like(rest_h, weight.options());
    auto ones = at::ones_like(rest_h, weight.options());    
    auto W1 = (ones - rest_h) * (ones - rest_w);
    auto W2 = rest_h * (ones - rest_w);
    auto W3 = (ones - rest_h) * rest_w;
    auto W4 = rest_h * rest_w;
    
    auto W1_Ph = -(ones - d_floor(P1, sigma, half_range_bot_h, half_range_top_h)) * (ones - rest_w) * weight;
    auto W2_Ph = -(d_floor(P1, sigma, half_range_bot_h, half_range_top_h) - ones) * (ones - rest_w) * weight;
    auto W3_Ph = -(ones - d_floor(P1, sigma, half_range_bot_h, half_range_top_h)) * rest_w * weight;
    auto W4_Ph = -(d_floor(P1, sigma, half_range_bot_h, half_range_top_h) - ones) * rest_w * weight;
    
    auto W1_Pw = -(ones - rest_h) * (ones - d_floor(P2, sigma, half_range_bot_w, half_range_top_w)) * weight;
    auto W2_Pw = -rest_h * (ones - d_floor(P2, sigma, half_range_bot_w, half_range_top_w)) * weight;
    auto W3_Pw = -(ones - rest_h)* (d_floor(P2, sigma, half_range_bot_w, half_range_top_w) - ones) * weight;
    auto W4_Pw = -rest_h * (d_floor(P2, sigma, half_range_bot_w, half_range_top_w) - ones) * weight;        
        
    const int height_out = dilation_h * (kernel_h - 1) + 1;
    const int width_out = dilation_w * (kernel_w - 1) + 1;

    
    const int num_kernels = channels_out * channels_in * kernel_h * kernel_w;    
    AT_DISPATCH_FLOATING_TYPES(weight.type(), "dcls_full_backward_cuda", [&] {
             
        interpolation_grad_kernel<scalar_t><<<GET_BLOCKS(num_kernels), 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                     num_kernels,
                                     grad_output.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                     weight.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
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
                                     weight.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
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
                                     weight.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
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
            1000.0*grad_P1/torch::norm(grad_P1),
            1000.0*grad_P2/torch::norm(grad_P2)};
}
