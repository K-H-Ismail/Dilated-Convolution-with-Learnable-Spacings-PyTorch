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

torch::Tensor d_sigmoid(torch::Tensor z, const double sigma) {
  auto s = torch::sigmoid(sigma * z);
  return (1.0 - s) * s * sigma;
}


torch::Tensor d_ceil(torch::Tensor z, const double sigma, const int bot, const int top) {
  auto s = torch::zeros_like(z);
  for (int i = 1-bot; i < top; i++) 
  { 
      s += d_sigmoid(z + static_cast<double>(i), sigma);
  }
  return s;
}

torch::Tensor d_floor(torch::Tensor z, const double sigma, const int bot, const int top) {
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
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> P_h,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> P_w,    
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> W1, 
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> W2,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> W3, 
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> W4,    
    const int ch_out, const int ch_in,
    const int kernel_h, const int kernel_w,
    const int height_out, const int width_out,    
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits>  interpolated_weight) {
  CUDA_KERNEL_LOOP(index, n) {
    int w_out = index % kernel_w;
    int h_out = (index / kernel_w) % kernel_h;
    int channel_in = (index / kernel_h / kernel_w) % ch_in;

      
    scalar_t w = weight[channel_in][h_out][w_out];
    int p_h = P_h[channel_in][h_out][w_out];
    int p_w = P_w[channel_in][h_out][w_out];
    int p_h_next = p_h + 1;
    int p_w_next = p_w + 1;      
   
    interpolated_weight[channel_in][p_h][p_w] += w * W1[channel_in][h_out][w_out];
    if(p_h_next < height_out) 
        interpolated_weight[channel_in][p_h_next][p_w] += w * W2[channel_in][h_out][w_out];
    if(p_w_next < width_out) 
        interpolated_weight[channel_in][p_h][p_w_next] += w * W3[channel_in][h_out][w_out];
    if(p_h_next < height_out & p_w_next < width_out) 
        interpolated_weight[channel_in][p_h_next][p_w_next] += w * W4[channel_in][h_out][w_out];    
      
  }
}

template <typename scalar_t>
__global__ void interpolation_grad_kernel(
    const int n,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_output,    
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> P_h,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> P_w,  
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> W1, 
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> W2,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> W3, 
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> W4,    
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> W1_Ph, 
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> W2_Ph,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> W3_Ph, 
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> W4_Ph,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> W1_Pw, 
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> W2_Pw,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> W3_Pw, 
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> W4_Pw,    
    const int ch_out, const int ch_in,
    const int kernel_h, const int kernel_w,
    const int height_out, const int width_out,     
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits>  interpolated_weight,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits>  interpolated_P1,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits>  interpolated_P2) {
  CUDA_KERNEL_LOOP(index, n) {
    int w_out = index % kernel_w;
    int h_out = (index / kernel_w) % kernel_h;
    int channel_in = (index / kernel_h / kernel_w) % ch_in;
      
    scalar_t w = weight[channel_in][h_out][w_out];
    int p_h = P_h[channel_in][h_out][w_out];
    int p_w = P_w[channel_in][h_out][w_out];
    int p_h_next = p_h + 1;
    int p_w_next = p_w + 1;       
      
    scalar_t g1 = grad_output[channel_in][p_h][p_w]; 
    scalar_t g2 = (p_h_next < height_out) ? grad_output[channel_in][p_h_next][p_w] : 0; 
    scalar_t g3 = (p_w_next < width_out) ? grad_output[channel_in][p_h][p_w_next] : 0; 
    scalar_t g4 = (p_h_next < height_out & p_w_next < width_out) ? grad_output[channel_in][p_h_next][p_w_next] : 0;       
    
    interpolated_weight[channel_in][h_out][w_out]+= g1 * W1[channel_in][h_out][w_out]
                                                                 + g2 * W2[channel_in][h_out][w_out]
                                                                 + g3 * W3[channel_in][h_out][w_out]
                                                                 + g4 * W4[channel_in][h_out][w_out];
      
    interpolated_P1[channel_in][h_out][w_out]+= g1 * w * W1_Ph[channel_in][h_out][w_out]
                                                                 + g2 * w * W2_Ph[channel_in][h_out][w_out]
                                                                 + g3 * w * W3_Ph[channel_in][h_out][w_out]
                                                                 + g4 * w * W4_Ph[channel_in][h_out][w_out];
      
    interpolated_P2[channel_in][h_out][w_out]+= g1 * w * W1_Pw[channel_in][h_out][w_out]
                                                                 + g2 * w * W2_Pw[channel_in][h_out][w_out]
                                                                 + g3 * w * W3_Pw[channel_in][h_out][w_out]
                                                                 + g4 * w * W4_Pw[channel_in][h_out][w_out];      
      
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
    
    const int num_kernels =  channels_in * kernel_h * kernel_w ;
    AT_DISPATCH_FLOATING_TYPES(weight.type(), "dcls_full_forward_cuda", [&] {
        for (int ch = 0; ch < channels_out ; ch++) {
                
            auto output_n = output.select(0, ch);
            auto weight_n = weight.select(0, ch);            
            auto P_h_n = P_h.select(0, ch);
            auto P_w_n = P_w.select(0, ch);
            auto W1_n = W1.select(0, ch);
            auto W2_n = W2.select(0, ch);
            auto W3_n = W3.select(0, ch);
            auto W4_n = W4.select(0, ch);            

            interpolation_kernel<scalar_t><<<GET_BLOCKS(num_kernels), 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                         num_kernels,
                                         weight_n.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                         P_h_n.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                         P_w_n.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                         W1_n.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                         W2_n.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                         W3_n.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                         W4_n.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                         channels_out, channels_in,
                                         kernel_h, kernel_w,
                                         height_out, width_out,
                                         output_n.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());

           output.select(0, ch) = output_n.view({channels_in, height_out, width_out}); 
            
        }
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
    
    const double sigma = 0.5;
    auto ones = at::ones_like(rest_h, weight.options());    
    auto W1 = (ones - rest_h) * (ones - rest_w);
    auto W2 = rest_h * (ones - rest_w);
    auto W3 = (ones - rest_h) * rest_w;
    auto W4 = rest_h * rest_w;
    
    auto W1_Ph = -(ones - d_floor(P1, sigma, half_range_bot_h, half_range_top_h)) * (ones - rest_w);
    auto W2_Ph = -(d_floor(P1, sigma, half_range_bot_h, half_range_top_h) - ones) * (ones - rest_w);
    auto W3_Ph = -(ones - d_floor(P1, sigma, half_range_bot_h, half_range_top_h)) * rest_w;
    auto W4_Ph = -(d_floor(P1, sigma, half_range_bot_h, half_range_top_h) - ones) * rest_w;
    
    auto W1_Pw = -(ones - rest_h) * (ones - d_floor(P2, sigma, half_range_bot_w, half_range_top_w));
    auto W2_Pw = -rest_h * (ones - d_floor(P2, sigma, half_range_bot_w, half_range_top_w));
    auto W3_Pw = -(ones - rest_h)* (d_floor(P2, sigma, half_range_bot_w, half_range_top_w) - ones);
    auto W4_Pw = -rest_h * (d_floor(P2, sigma, half_range_bot_w, half_range_top_w) - ones);        
        
    const int height_out = dilation_h * (kernel_h - 1) + 1;
    const int width_out = dilation_w * (kernel_w - 1) + 1;
    
    const int num_kernels = channels_in * kernel_h * kernel_w;    
    AT_DISPATCH_FLOATING_TYPES(weight.type(), "dcls_full_backward_cuda", [&] {
        for (int ch = 0; ch < channels_out ; ch++) {

            auto grad_output_n = grad_output.select(0, ch);
            auto weight_n = weight.select(0, ch);            
            auto P_h_n = P_h.select(0, ch);
            auto P_w_n = P_w.select(0, ch);
            auto W1_n = W1.select(0, ch);
            auto W2_n = W2.select(0, ch);
            auto W3_n = W3.select(0, ch);
            auto W4_n = W4.select(0, ch);  
            auto W1_Ph_n = W1_Ph.select(0, ch);
            auto W2_Ph_n = W2_Ph.select(0, ch);
            auto W3_Ph_n = W3_Ph.select(0, ch);
            auto W4_Ph_n = W4_Ph.select(0, ch);  
            auto W1_Pw_n = W1_Pw.select(0, ch);
            auto W2_Pw_n = W2_Pw.select(0, ch);
            auto W3_Pw_n = W3_Pw.select(0, ch);
            auto W4_Pw_n = W4_Pw.select(0, ch);
            auto grad_weight_n = grad_weight.select(0, ch);
            auto grad_P1_n = grad_P1.select(0, ch); 
            auto grad_P2_n = grad_P2.select(0, ch);             
            
            interpolation_grad_kernel<scalar_t><<<GET_BLOCKS(num_kernels), 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                         num_kernels,
                                         grad_output_n.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                         weight_n.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                         P_h_n.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                         P_w_n.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                         W1_n.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                         W2_n.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                         W3_n.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                         W4_n.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                         W1_Ph_n.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                         W2_Ph_n.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                         W3_Ph_n.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                         W4_Ph_n.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                         W1_Pw_n.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                         W2_Pw_n.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                         W3_Pw_n.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                         W4_Pw_n.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                         channels_out, channels_in,
                                         kernel_h, kernel_w, 
                                         height_out, width_out,                                 
                                         grad_weight_n.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                         grad_P1_n.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                         grad_P2_n.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
                                         
            grad_weight.select(0, ch) = grad_weight_n.view({channels_in, kernel_h, kernel_w});
            grad_P1.select(0, ch) = grad_P1_n.view({channels_in, kernel_h, kernel_w});
            grad_P2.select(0, ch) = grad_P2_n.view({channels_in, kernel_h, kernel_w});
        }
    });
    

    return {grad_weight,
            grad_P1,
            grad_P2};
}
