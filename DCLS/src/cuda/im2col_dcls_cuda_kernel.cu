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

template <typename scalar_t>
__global__ void im2col_dcls_kernel(
    const int n,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> P_h, 
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> P_w,
    const int height_in, const int width_in,
    const int ch_in, const int ch_out,
    const int kernel_h, const int kernel_w,
    const int height_out, const int width_out,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int shift_h, const int shift_w,    
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    int w_out = index % width_out;
    int h_out = (index / width_out) % height_out;
    int channel_in = (index / height_out / width_out) % ch_in;
    int channel_out = (index / height_out / width_out / ch_in) % ch_out;
      
    int h_in = h_out * stride_h - pad_h;
    int w_in = w_out * stride_w - pad_w;

    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        int l_dilation_h = static_cast<int>(P_h[channel_out][channel_in][i][j]) ;//i * dilation_h;
        int l_dilation_w = static_cast<int>(P_w[channel_out][channel_in][i][j]) ;//j * dilation_w;
          
        int h = h_in + l_dilation_h + shift_h;
        int w = w_in + l_dilation_w + shift_w;
          
        if (h >= 0 && w >= 0 && h < height_in && w < width_in) {
            data_col[channel_out][(channel_in*kernel_h + i)*kernel_w + j][h_out*width_out + w_out] = 
                input[channel_in][h][w];
        }       

      }
    }
  }
}

torch::Tensor  einsum(
    torch::Tensor weight,
    torch::Tensor columns) {
    
    return at::einsum("ij, ijk -> ik",{weight, columns});
}

torch::Tensor  im2col_dcls_cuda(
    torch::Tensor im,
    torch::Tensor P_h, torch::Tensor P_w,     
    const int dilation_h, const int dilation_w, 
    const int padding_h, const int padding_w,
    const int stride_h, const int stride_w,
    const int height_out, const int width_out,    
    const int shift_h, const int shift_w) {
    
    const int height = im.size(1);
    const int width = im.size(2);
    
    const int channels_out = P_h.size(0);
    const int channels_in = P_h.size(1);
    const int kernel_h = P_h.size(2);
    const int kernel_w = P_h.size(3);    
    

    auto columns = at::zeros({channels_out, channels_in * kernel_h * kernel_w, height_out * width_out}, im.options());
    
    const int num_kernels = channels_out * channels_in * height_out * width_out;
    
    AT_DISPATCH_FLOATING_TYPES(im.type(), "im2col_dcls_cuda", [&] {

        im2col_dcls_kernel<scalar_t><<<GET_BLOCKS(num_kernels), 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                         num_kernels,
                                         im.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                         P_h.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                         P_w.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                         height, width,
                                         channels_in, channels_out,
                                         kernel_h, kernel_w, 
                                         height_out, width_out,
                                         padding_h, padding_w, 
                                         stride_h, stride_w, 
                                         dilation_h, dilation_w,
                                         shift_h,shift_w,
                                         columns.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
    });
    
    return columns;

}

torch::Tensor  einsum_dcls_forward(
    torch::Tensor im,
    torch::Tensor weights,    
    torch::Tensor P_h, torch::Tensor P_w,     
    const int dilation_h, const int dilation_w, 
    const int padding_h, const int padding_w,
    const int stride_h, const int stride_w,
    const int height_out, const int width_out) {
    
    const int channels_out = weights[0].size(0);
    const int channels_in = weights[0].size(1);
    const int kernel_h = weights[0].size(2);
    const int kernel_w = weights[0].size(3); 
    
    auto output = at::zeros({channels_out, height_out * width_out}, im.options());
   
    for (int shift_h = 0; shift_h < 2; ++shift_h) {
        for (int shift_w = 0; shift_w < 2; ++shift_w) {
            auto columns = im2col_dcls_cuda(im, P_h, P_w, dilation_h, dilation_w, padding_h, padding_w, 
                                       stride_h, stride_w, height_out, width_out, shift_h, shift_w);
            output += einsum(weights.select(0,shift_h + 2 * shift_w).view({channels_out, channels_in*kernel_h*kernel_w}), columns);
        }
    }

    return output; 
}

torch::Tensor  chunked_einsum_dcls_forward(
    torch::Tensor im,
    torch::Tensor weights,    
    torch::Tensor P_h, torch::Tensor P_w,     
    const int dilation_h, const int dilation_w, 
    const int padding_h, const int padding_w,
    const int stride_h, const int stride_w,
    const int height_out, const int width_out,
    const int chunk_size) {
    
    const int channels_out = weights.select(0,0).size(0);
    const int channels_in = weights.select(0,0).size(1);
    const int kernel_h = weights.select(0,0).size(2);
    const int kernel_w = weights.select(0,0).size(3); 
    
    int nb_chunks = (channels_out-1)/chunk_size + 1; 

    auto output = at::zeros({}, im.options());
    
    auto chunked_P_h = P_h.chunk(nb_chunks,0);
    auto chunked_P_w = P_w.chunk(nb_chunks,0);

    auto chunked_weights = weights.chunk(nb_chunks,1);
    
    for (int chunk = 0; chunk < nb_chunks; ++chunk) {
        
        auto P_h_chunk = chunked_P_h[chunk];
        auto P_w_chunk = chunked_P_w[chunk];
        auto weights_chunk = chunked_weights[chunk];
        
        auto output_chunk = einsum_dcls_forward(im, weights_chunk, P_h_chunk, P_w_chunk, dilation_h, dilation_w, 
                                   padding_h, padding_w, stride_h, stride_w, height_out, width_out);
        
        output = chunk == 0 ? output_chunk : at::cat({output,output_chunk},0);
    }
        
    return output; 
}


torch::Tensor  einsum_dcls_backward(
    torch::Tensor im,
    torch::Tensor weights,
    torch::Tensor grad,    
    torch::Tensor P_h, torch::Tensor P_w,     
    const int dilation_h, const int dilation_w, 
    const int padding_h, const int padding_w,
    const int stride_h, const int stride_w,
    const int height_out, const int width_out) {
    
    const int channels_out = weights[0].size(0);
    const int channels_in = weights[0].size(1);
    const int kernel_h = weights[0].size(2);
    const int kernel_w = weights[0].size(3); 
    
    auto output = at::zeros({channels_out, channels_in * kernel_h * kernel_w}, im.options());
    auto columns = at::zeros({channels_out, channels_in * kernel_h * kernel_w, height_out * width_out}, im.options());    
   
    for (int shift_h = 0; shift_h < 2; ++shift_h) {
        for (int shift_w = 0; shift_w < 2; ++shift_w) {
            auto columns_tmp = im2col_dcls_cuda(im, P_h, P_w, dilation_h, dilation_w, padding_h, padding_w, 
                                       stride_h, stride_w, height_out, width_out, shift_h, shift_w);
            columns += weights.select(0,shift_h + 2 * shift_w)
                             .view({channels_out, channels_in*kernel_h*kernel_w})
                             .unsqueeze(-1) * columns_tmp;
        }
    }
    
    output = einsum(grad, columns.permute({0,2,1}));
    
    return output; 
}

torch::Tensor  chunked_einsum_dcls_backward(
    torch::Tensor im,
    torch::Tensor weights,
    torch::Tensor grad,    
    torch::Tensor P_h, torch::Tensor P_w,     
    const int dilation_h, const int dilation_w, 
    const int padding_h, const int padding_w,
    const int stride_h, const int stride_w,
    const int height_out, const int width_out,
    const int chunk_size) {
    
    const int channels_out = weights.select(0,0).size(0);
    const int channels_in = weights.select(0,0).size(1);
    const int kernel_h = weights.select(0,0).size(2);
    const int kernel_w = weights.select(0,0).size(3); 
    
    int nb_chunks = (channels_out-1)/chunk_size + 1; 

    auto output = at::zeros({}, im.options());
    
    auto chunked_P_h = P_h.chunk(nb_chunks,0);
    auto chunked_P_w = P_w.chunk(nb_chunks,0);

    auto chunked_weights = weights.chunk(nb_chunks,1);
    auto chunked_grad = grad.chunk(nb_chunks,0);    
    
    for (int chunk = 0; chunk < nb_chunks; ++chunk) {
        
        auto P_h_chunk = chunked_P_h[chunk];
        auto P_w_chunk = chunked_P_w[chunk];
        auto weights_chunk = chunked_weights[chunk];
        auto grad_chunk = chunked_grad[chunk];        
        
        auto output_chunk = einsum_dcls_backward(im, weights_chunk, grad_chunk, P_h_chunk, P_w_chunk, dilation_h, dilation_w, 
                                   padding_h, padding_w, stride_h, stride_w, height_out, width_out);
        
        output = chunk == 0 ? output_chunk : at::cat({output,output_chunk},0);
    }
        
    return output; 
}