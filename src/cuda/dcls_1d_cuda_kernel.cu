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
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> weight,
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
      
    scalar_t w = weight[channel_out][channel_in][l_out];
    int p = P[channel_out][channel_in][l_out];

    int p_next = p + 1;     
   
    interpolated_weight[channel_out][channel_in][p] += w * W1[channel_out][channel_in][l_out];
    if(p_next < length_out) 
        interpolated_weight[channel_out][channel_in][p_next] += w * W2[channel_out][channel_in][l_out]; 
  }
}

template <typename scalar_t>
__global__ void interpolation_grad_kernel(
    const int n,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_output,    
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> weight,
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
      
    scalar_t w = weight[channel_out][channel_in][l_out];
    int p = P[channel_out][channel_in][l_out];

    int p_next = p + 1;       
      
    scalar_t g1 = grad_output[channel_out][channel_in][p]; 
    scalar_t g2 = (p_next < length_out) ? grad_output[channel_out][channel_in][p_next] : 0; 
      
    
    interpolated_weight[channel_out][channel_in][l_out] += g1 * W1[channel_out][channel_in][l_out]
                                                        +  g2 * W2[channel_out][channel_in][l_out];
      
  }
}

torch::Tensor  dcls_1d_cuda_forward(  
    torch::Tensor weight,
    torch::Tensor P1,
    const int dilation
    ) {
    
    const int channels_out = weight.size(0);
    const int channels_in = weight.size(1);    
    const int kernel = weight.size(2);
 

    const int half_range_bot = dilation*kernel/2;
    const int half_range_top = half_range_bot - (dilation*kernel + 1)%2;

    
    auto P = at::clamp(at::floor(P1),-half_range_bot,half_range_top);
    auto rest = at::clamp(P1,-half_range_bot,half_range_top) - P;
    
    P += dilation*kernel/2 ;

    auto ones = at::ones_like(rest, weight.options());    
    auto W1 = ones - rest;
    auto W2 = rest;   
    
    const int length_out = dilation * (kernel - 1) + 1;
   
    auto output = torch::zeros({channels_out, channels_in, length_out}, weight.options());
    
    const int num_kernels =  channels_out * channels_in * kernel;
    AT_DISPATCH_FLOATING_TYPES(weight.type(), "dcls_1d_forward_cuda", [&] {
          
        interpolation_kernel<scalar_t><<<GET_BLOCKS(num_kernels), 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                     num_kernels,
                                     weight.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
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

std::vector<torch::Tensor> dcls_1d_cuda_backward(   
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
    
    auto P = at::clamp(at::floor(P1),-half_range_bot,half_range_top);
    auto rest = at::clamp(P1,-half_range_bot,half_range_top) - P;
        
    
    P += dilation*kernel/2;
   
    
    auto sigma = 0.5*at::ones_like(rest, weight.options());
    auto ones = at::ones_like(rest, weight.options());    
    auto W1 = ones - rest;
    auto W2 = rest;
    
    auto W1_P = (d_floor(P1, sigma, half_range_bot, half_range_top) - ones) * weight;
    auto W2_P = - W1_P;
    
    const int length_out = dilation * (kernel - 1) + 1;

    const int num_kernels = channels_out * channels_in * kernel;    
    AT_DISPATCH_FLOATING_TYPES(weight.type(), "dcls_1d_backward_cuda", [&] {
             
        interpolation_grad_kernel<scalar_t><<<GET_BLOCKS(num_kernels), 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                     num_kernels,
                                     grad_output.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     weight.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
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
                                     weight.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     P.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     W1_P.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     W2_P.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     channels_out, channels_in,
                                     kernel, 
                                     length_out,                                     
                                     grad_P1.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
        
    });
    auto norm_w = torch::norm(weight);
    auto norm_grad_w = torch::norm(grad_weight);

    return {grad_weight,
            dilation*grad_P1/torch::norm(grad_P1)};
}
