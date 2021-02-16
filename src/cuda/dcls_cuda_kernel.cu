#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <vector>

#define CUDA_KERNEL_LOOP(i, n)                          \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;   \
      i < (n);                                          \
      i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
inline int GET_BLOCKS(const int N)
{
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

template <typename scalar_t>
__device__ scalar_t dmcn_im2col_bilinear(const scalar_t *bottom_data, const int data_width,
                                      const int height, const int width, scalar_t h, scalar_t w)
{
  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  scalar_t lh = h - h_low;
  scalar_t lw = w - w_low;
  scalar_t hh = 1 - lh, hw = 1 - lw;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0)
    v1 = bottom_data[h_low * data_width + w_low];
  scalar_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
    v2 = bottom_data[h_low * data_width + w_high];
  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = bottom_data[h_high * data_width + w_low];
  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = bottom_data[h_high * data_width + w_high];

  scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <typename scalar_t>
__global__ void dcls_cuda_forward_kernel(const int n,
                                         const scalar_t *data_im, 
                                         const scalar_t *P1, const scalar_t *rest1,
                                         const scalar_t *P2, const scalar_t *rest2,
                                         const int batch_size, const int num_channels, const int height_col, const int width_col,
                                         const int height, const int width, const int kernel_h, const int kernel_w,
                                         const int pad_h, const int pad_w,
                                         const int stride_h, const int stride_w,
                                         const int dilation_h, const int dilation_w,                                         
                                         scalar_t *data_col)
{
  // launch channels * batch_size * height_col * width_col cores
  CUDA_KERNEL_LOOP(index, n)
  {
    // NOTE(CharlesShang): different from Dai Jifeng's MXNet implementation, col_buffer is of shape (c*kw*kh, N, oh, ow)
    // here columns is of shape (N, c*kw*kh, oh * ow), need to adapt axis
    // NOTE(Jiarui XU): different from CharlesShang's implementation, col_buffer is of shape (N, c*kw*kh, oh * ow)
    // here columns is of shape (c*kw*kh, N, oh, ow), need to adapt axis

    // index index of output matrix
    const int w_col = index % width_col;
    const int h_col = (index / width_col) % height_col;
    const int b_col = (index / width_col / height_col) % batch_size;
    const int c_im = (index / width_col / height_col) / batch_size;
    const int c_col = c_im * kernel_h * kernel_w;


    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;

     scalar_t *data_col_ptr = data_col + ((c_col * batch_size + b_col) * height_col + h_col) * width_col + w_col;

    const scalar_t *data_im_ptr = data_im + (b_col * num_channels + c_im) * height * width;
    //todo: const scalar_t *data_offset_ptr = data_offset + (b_col * deformable_group + deformable_group_index) * 2 * kernel_h * kernel_w * height_col * width_col;

    for (int i = 0; i < kernel_h; ++i)
    {
      for (int j = 0; j < kernel_w; ++j)
      {
        //todo: const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
        //todo: const int data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col + w_col;
        //todo: const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
        //todo: const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];
        scalar_t val = static_cast<scalar_t>(0);
        const scalar_t h_im = h_in + i * dilation_h /*+ offset_h*/;
        const scalar_t w_im = w_in + j * dilation_w /*+ offset_w*/;
        if (h_im > -1 && w_im > -1 && h_im < height && w_im < width)
        {

          val = dmcn_im2col_bilinear(data_im_ptr, width, height, width, h_im, w_im);
        }
        *data_col_ptr = val;
        data_col_ptr += batch_size * height_col * width_col;
      }
    }
  }
}


torch::Tensor  dcls_cuda_forward(
    torch::Tensor input,    
    torch::Tensor weight,
    torch::Tensor P1,
    torch::Tensor rest1,
    torch::Tensor P2,
    torch::Tensor rest2,
    torch::Tensor bias,
    const int channel_out, const int channel_in,
    const int kernel_h, const int kernel_w,
    const int dilation_h, const int dilation_w, 
    const int stride_h, const int stride_w, 
    const int padding_h, const int padding_w, 
    const int groups) {
    
    const int batch = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    const int channels_out = weight.size(0);
    const int channels_in = weight.size(1);
    const int kernel_h_ = weight.size(2);
    const int kernel_w_ = weight.size(3);
    
    const int height_out = (height + 2 * padding_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width + 2 * padding_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    
    auto output = torch::empty({batch * height_out * width_out, channels_out}, input.options());
    
    // prepare group weight and bias
    auto weight_g = weight.view({groups, channels_out/groups, channels_in, kernel_h, kernel_w});
    auto bias_g = bias.view({groups, channels_out/groups});
    
    
    const int num_kernels = channels * batch * height * width;
    auto columns = at::empty({channels * kernel_h * kernel_w, batch * height_out * width_out}, input.options());
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "dcls_forward_cuda", ([&] {
    dcls_cuda_forward_kernel<scalar_t><<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                                     num_kernels,
                                     input.data<scalar_t>(),
                                     P1.data<scalar_t>() ,
                                     rest1.data<scalar_t>() ,
                                     P2.data<scalar_t>() ,
                                     rest2.data<scalar_t>() ,
                                     batch, channels, height, width,
                                     height_out, width_out, kernel_h, kernel_w,
                                     padding_h, padding_w, stride_h, stride_w, dilation_h, dilation_w,
                                     columns.data<scalar_t>());
    }));
    
    auto columns_g = columns.view({groups, channels/groups * kernel_h * kernel_w, batch * height_out * width_out});
    auto output_g = output.view({batch * height_out * width_out, groups, channels_out/groups});
    for (int g = 0; g < groups; ++g)
    {
        auto columns_gm = columns_g.select(0, g).t();
        auto weight_gm = weight_g.select(0, g).view({channels_out/groups, channels_in * kernel_h * kernel_w}).t();
        auto output_m = at::addmm(bias_g.select(0, g), columns_gm, weight_gm);
        output_g.select(1, g) = output_m.view({batch * height_out * width_out, channels_out/groups});
    }
        
    output = output.view({batch, height_out, width_out, channels_out}).permute({0, 3, 1, 2}).contiguous();    
    return output;
}

std::vector<torch::Tensor> dcls_cuda_backward(
    torch::Tensor input,    
    torch::Tensor weight,
    torch::Tensor P1,
    torch::Tensor rest1,
    torch::Tensor P2,
    torch::Tensor rest2,
    torch::Tensor grad_output,      
    torch::Tensor bias,
    const int channel_out, const int channel_in,
    const int kernel_h, const int kernel_w,
    const int dilation_h, const int dilation_w, 
    const int stride_h, const int stride_w, 
    const int padding_h, const int padding_w, 
    const int groups) {
    
    auto grad_input = torch::zeros_like(input);      
    auto grad_weight = torch::zeros_like(weight);
    auto grad_P1 = torch::zeros_like(P1);
    auto grad_P2 = torch::zeros_like(P2);
    auto grad_bias = torch::zeros_like(bias);
    
    return {grad_input,
            grad_weight,
            grad_P1,
            grad_P2,
            grad_bias};
}




