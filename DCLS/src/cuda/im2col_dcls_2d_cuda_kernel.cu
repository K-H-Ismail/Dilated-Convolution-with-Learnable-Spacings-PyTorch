#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <math.h>
#include <vector>
#include <assert.h>

#include <stdio.h>


#define CUDA_KERNEL_LOOP(i, n)                                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);                 \
       i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
const int kMaxGridNum = 65535;
inline int GET_BLOCKS(const int N) {
  return std::min(kMaxGridNum, (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);
}

inline long GET_FREE_MEMORY() {
  size_t free, total;
  cudaMemGetInfo(&free, &total);
  return static_cast<long>(free) > 0 ? static_cast<long>(free) : 1;
}

namespace constants {
    const long GLOBAL_FREE_MEMORY = GET_FREE_MEMORY();
}

// col2im kernel, identical to the one in CuDNN 
template <typename scalar_t>
__global__ void col2im_kernel(
    const int n,
    const scalar_t* data_col, 
    const int height,
    const int width,
    const int channels,
    const int kernel_h,
    const int kernel_w,
    const int pad_height,
    const int pad_width,
    const int stride_height,
    const int stride_width,
    const int dilation_height,
    const int dilation_width,
    const int height_col,
    const int width_col,
    scalar_t* data_im) {
  CUDA_KERNEL_LOOP(index, n) {
    scalar_t val = static_cast<scalar_t>(0);
    const int w_im = index % width + pad_width;
    const int h_im = (index / width) % height + pad_height;
    const int c_im = (index / (width * height)) % channels;
    const int b_im = index / (width * height * channels);
    int kernel_extent_w = (kernel_w - 1) * dilation_width + 1;
    int kernel_extent_h = (kernel_h - 1) * dilation_height + 1;
    // compute the start and end of the output
    const int w_col_start = (w_im < kernel_extent_w)
        ? 0
        : (w_im - kernel_extent_w) / stride_width + 1;
    const int w_col_end = ::min(w_im / stride_width + 1, width_col);
    const int h_col_start = (h_im < kernel_extent_h)
        ? 0
        : (h_im - kernel_extent_h) / stride_height + 1;
    const int h_col_end = ::min(h_im / stride_height + 1, height_col);

    // TODO: use LCM of stride and dilation to avoid unnecessary loops
    for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
      for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
        int h_k = (h_im - h_col * stride_height);
        int w_k = (w_im - w_col * stride_width);
        if (h_k % dilation_height == 0 && w_k % dilation_width == 0) {
          h_k /= dilation_height;
          w_k /= dilation_width;
          int data_col_index =
              ((((b_im * channels + c_im) * kernel_h + h_k) * kernel_w + w_k) * height_col +
               h_col) * width_col + w_col;
          val += data_col[data_col_index];
        }
      }
    }
    data_im[index] = static_cast<scalar_t>(val);
  }
}

// Adaptation of im2col kernel to dcls case 
template <typename scalar_t>
__global__ void im2col_dcls_2d_batch_kernel(
    const int n,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> P_h, 
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> P_w,
    const int height_in, const int width_in,
    const int ch_in, const int batch_size,
    const int kernel_h, const int kernel_w,
    const int height_out, const int width_out,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int groups,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    int w_out = index % width_out;
    int h_out = (index / width_out) % height_out;
    int channel_in = (index / height_out / width_out) % ch_in;
    int group = (index / height_out / width_out / ch_in) % groups;
    int batch = (index / height_out / width_out / ch_in / groups) % batch_size;
      
    int h_in = h_out * stride_h - pad_h;
    int w_in = w_out * stride_w - pad_w;

    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        int l_dilation_h = static_cast<int>(P_h[channel_in][i][j]) ;//i * dilation_h in the standard case;
        int l_dilation_w = static_cast<int>(P_w[channel_in][i][j]) ;//j * dilation_w in the standard case;
        
        // Shifts are used for interpolation
        for (int shift_h = 0; shift_h < 2; ++shift_h) {
            for (int shift_w = 0; shift_w < 2; ++shift_w) {
                int h = h_in + l_dilation_h + shift_h;
                int w = w_in + l_dilation_w + shift_w;

                if (h >= 0 && w >= 0 && h < height_in && w < width_in) {
                    data_col[batch][group][shift_h + 2 * shift_w][(channel_in * kernel_h + i) * kernel_w + j]
                            [h_out * width_out + w_out] = input[batch][group][channel_in][h][w];
                }
            }
        }       

      }
    }
  }
}

// Wrapper for im2col dcls kernel
torch::Tensor  im2col_dcls_2d_batch_cuda(
    torch::Tensor im,
    torch::Tensor P_h, torch::Tensor P_w,     
    const int dilation_h, const int dilation_w, 
    const int padding_h, const int padding_w,
    const int stride_h, const int stride_w,
    const int height_out, const int width_out) {
    
    const int batch_size = im.size(0);
    const int groups = im.size(1);
    const int height = im.size(3);
    const int width = im.size(4);
    
    const int channels_in = P_h.size(0);
    const int kernel_h = P_h.size(1);
    const int kernel_w = P_h.size(2);    
    
    // The output is a batch_size x groups x 4 x channels_in * kernel_h * kernel_w x height_out * width_out tensor
    // as the interpolation produces 2 x 2 more ouputs  
    auto columns = at::zeros({batch_size, groups, 4, channels_in * kernel_h * kernel_w, height_out * width_out}, im.options());
    
    const int num_kernels = batch_size * channels_in * groups * height_out * width_out;
    
    AT_DISPATCH_FLOATING_TYPES(im.type(), "im2col_dcls_2d_batch_cuda", [&] {

        im2col_dcls_2d_batch_kernel<scalar_t><<<GET_BLOCKS(num_kernels), 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                         num_kernels,
                                         im.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                                         P_h.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                         P_w.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                         height, width,
                                         channels_in, batch_size,
                                         kernel_h, kernel_w, 
                                         height_out, width_out,
                                         padding_h, padding_w, 
                                         stride_h, stride_w, 
                                         dilation_h, dilation_w,
                                         groups,
                                         columns.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>());
    });
    
    return columns;
}

torch::Tensor  mm_dcls_2d_forward(
    torch::Tensor im,
    torch::Tensor weights,    
    torch::Tensor P_h, torch::Tensor P_w,     
    const int dilation_h, const int dilation_w, 
    const int padding_h, const int padding_w,
    const int stride_h, const int stride_w,
    const int height_out, const int width_out) {
    
    const int batch_size = im.size(0);
    const int groups = im.size(1);
    
    // Weights contains the interpolated weights 
    // weights tensor has a size of groups x channels_out x 4 x channels_in x kernel_h x kernel_w
    const int channels_out = weights.select(2,0).size(1);
    const int channels_in = weights.select(2,0).size(2);
    const int kernel_h = weights.select(2,0).size(3);
    const int kernel_w = weights.select(2,0).size(4); 
    
    auto output = at::zeros({batch_size, groups, channels_out, height_out * width_out}, im.options());
   
    // Call im2col dcls
    auto columns = im2col_dcls_2d_batch_cuda(im, P_h, P_w, dilation_h, dilation_w, padding_h, padding_w, 
                                          stride_h, stride_w, height_out, width_out);
    
    // Apply matrix-matrix multiplication
    output = at::matmul(weights.view({groups, channels_out, 4 * channels_in * kernel_h * kernel_w}), 
                        columns.view({batch_size, groups, 4 * channels_in * kernel_h * kernel_w, height_out * width_out}));

    return output; 
}

std::vector<torch::Tensor> mm_dcls_2d_backward(
    torch::Tensor im,
    torch::Tensor weights,
    torch::Tensor weights_Ph,
    torch::Tensor weights_Pw,    
    torch::Tensor grad,    
    torch::Tensor P_h, torch::Tensor P_w,     
    const int dilation_h, const int dilation_w, 
    const int padding_h, const int padding_w,
    const int stride_h, const int stride_w,
    const int height_out, const int width_out) {
    
    const int batch_size = im.size(0);
    const int groups = im.size(1);    
    
    // Weights contains the interpolated weights 
    // weights tensor has a size of groups x 4 x channels_out x channels_in x kernel_h x kernel_w    
    const int channels_out = weights.select(1,0).size(1);
    const int channels_in = P_h.size(0);
    const int kernel_h = P_h.size(1);
    const int kernel_w = P_h.size(2); 
    
    // grad_weights is a groups x 4 x channels_out x channels_in * kernel_h * kernel_w  tensor
    // this intermediate variable will serve to calculate grads with respect to weights, Ph and Pw       
   
    // Call im2col dcls
    auto columns = im2col_dcls_2d_batch_cuda(im, P_h, P_w, dilation_h, dilation_w, padding_h, padding_w, 
                                             stride_h, stride_w, height_out, width_out);

    // Apply matrix-matrix multiplication
    auto grad_weights = (at::matmul(grad.unsqueeze(2), columns.permute({0, 1, 2, 4, 3}))).sum(0);

    // Sum over interpolations and groups, and take mean value over channels_out of positions
    auto grad_weight = (grad_weights * weights.view({groups, 4, channels_out, channels_in * kernel_h * kernel_w})).sum(1);
    auto grad_Ph = (grad_weights * weights_Ph.view({groups, 4, channels_out, channels_in * kernel_h * kernel_w}))
                                                                                                  .sum(1).sum(0).mean(0);
    auto grad_Pw = (grad_weights * weights_Pw.view({groups, 4, channels_out, channels_in * kernel_h * kernel_w}))
                                                                                                  .sum(1).sum(0).mean(0);
    
    return {grad_weight,
            grad_Ph,
            grad_Pw};
    
}
