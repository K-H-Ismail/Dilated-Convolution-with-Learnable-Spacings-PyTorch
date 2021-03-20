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


/*template <typename scalar_t>
__global__ void gather_kernel(
    const int n,
    scalar_t* grad_weight,
    const int ch_in, const int ch_out,
    const int kernel_h, const int kernel_w,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> gather_weight) {
  CUDA_KERNEL_LOOP(index, n) {
    int w_out = index % kernel_w;
    int h_out = (index / kernel_w) % kernel_h;
    int channel_in = (index / kernel_h / kernel_w) % ch_in;
    int channel_out = (index / kernel_h / kernel_w / ch_in) % ch_out;
    
    scalar_t* w_val = grad_weight + ((channel_out * ch_in + channel_in) * kernel_h + h_out) * kernel_w + w_out;
      
    //scalar_t* col = gather_weight + ((channel_out * ch_in + channel_in) * kernel_h + h_out) * kernel_w + w_out;

    gather_weight[channel_out][channel_in][h_out][w_out] = *(w_val+ kernel_h * kernel_w * (3*channel_in + 3*ch_in*channel_out));
  }
}*/

template <typename scalar_t>
__global__ void interpolation_kernel(
    const int n,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> W1, 
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> W2,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> W3, 
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> W4,
    const int ch_in, const int ch_out,
    const int kernel_h, const int kernel_w,
    scalar_t* interpolated_weight) {
  CUDA_KERNEL_LOOP(index, n) {
    int w_out = index % kernel_w;
    int h_out = (index / kernel_w) % kernel_h;
    int channel_in = (index / kernel_h / kernel_w) % ch_in;
    int channel_out = (index / kernel_h / kernel_w / ch_in) % ch_out;
    
    scalar_t w_val = weight[channel_out][channel_in][h_out][w_out];
      
    scalar_t* col = interpolated_weight + ((channel_out * ch_in + channel_in) * kernel_h + h_out) * kernel_w + w_out;

    
    *(col + kernel_h * kernel_w * (3*channel_in + 3*ch_in*channel_out)) = w_val * W1[channel_in][h_out][w_out];
    *(col + kernel_h * kernel_w * (3*channel_in + 1 + 3*ch_in*channel_out)) = w_val * W2[channel_in][h_out][w_out];
    *(col + kernel_h * kernel_w * (3*channel_in + 2 + 3*ch_in*channel_out)) = w_val * W3[channel_in][h_out][w_out];
    *(col + kernel_h * kernel_w * (3*channel_in + 3 + 3*ch_in*channel_out)) = w_val * W4[channel_in][h_out][w_out];


  }
}

template <typename scalar_t>
__global__ void im2col_kernel_old(
    const int n,
    const scalar_t* input, 
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> P_h, 
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> P_w,
    const int height_in, const int width_in,
    const int ch_in, const int ch_out,
    const int kernel_h, const int kernel_w,
    const int height_out, const int width_out,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int groups,    
    scalar_t* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    int w_out = index % width_out;

    int idx = index / width_out;

    int h_out = idx % height_out;
    int channel_in = (idx / height_out)% ch_in;
    int channel_out = channel_in * kernel_h * kernel_w  ;
    int h_in = h_out * stride_h - pad_h;
    int w_in = w_out * stride_w - pad_w;

    scalar_t* col = data_col + (channel_out * height_out + h_out) * width_out + w_out;
    const scalar_t* im = input + (channel_in * height_in + h_in) * width_in + w_in;

    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        int l_dilation_h = static_cast<int>(P_h[channel_in/groups][i][j]) ;//i * dilation_h;
        int l_dilation_w = static_cast<int>(P_w[channel_in/groups][i][j]) ;//j * dilation_w;
          
        int h = h_in + l_dilation_h;
        int w = w_in + l_dilation_w;
          
        if (h >= 0 && w >= 0 && h < height_in && w < width_in) {
            scalar_t im_val = im[l_dilation_h * width_in + l_dilation_w];
            *(col) = im_val;

        }       
        else {
            *col = static_cast<scalar_t>(0);
        }

        col += height_out * width_out;


      }
    }
  }
}
template <typename scalar_t>
__global__ void im2col_kernel(
    const int n,
    const scalar_t* input, 
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> P_h, 
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> P_w,
    const int height_in, const int width_in,
    const int ch_in, const int ch_out,
    const int kernel_h, const int kernel_w,
    const int height_out, const int width_out,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int groups,    
    scalar_t* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    int w_out = index % width_out;

    int idx = index / width_out;

    int h_out = idx % height_out;
    int channel_in = (idx / height_out)% ch_in;
    int channel_out = channel_in * kernel_h * kernel_w  ;
    int h_in = h_out * stride_h - pad_h;
    int w_in = w_out * stride_w - pad_w;

    scalar_t* col = data_col + (channel_out * height_out + h_out) * width_out + w_out;
    const scalar_t* im = input + (channel_in * height_in + h_in) * width_in + w_in;

    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        int l_dilation_h = static_cast<int>(P_h[channel_in/groups][i][j]);
        int l_dilation_w = static_cast<int>(P_w[channel_in/groups][i][j]);
          
        int h = h_in + l_dilation_h;
        int w = w_in + l_dilation_w;
          

        *(col + height_out * width_out * kernel_h * kernel_w * 3*channel_in) = 
            (h >= 0 && w >= 0 && h < height_in && w < width_in) ? 
                im[l_dilation_h * width_in + l_dilation_w] : 
                  *(col + height_out * width_out * kernel_h * kernel_w * 3*channel_in) = static_cast<scalar_t>(0);
          
        *(col + height_out * width_out * kernel_h * kernel_w * (3*channel_in+1)) = 
            (h >= -1 && w >= 0 && h < height_in - 1 && w < width_in) ? 
                im[(l_dilation_h+1) * width_in + l_dilation_w] : 
                  *(col + height_out * width_out * kernel_h * kernel_w * (3*channel_in+1)) = static_cast<scalar_t>(0);
          
       *(col + height_out * width_out * kernel_h * kernel_w * (3*channel_in+2)) = 
            (h >= 0 && w >= -1 && h < height_in && w < width_in - 1) ? 
                im[l_dilation_h * width_in + (l_dilation_w+1)] : 
                  *(col + height_out * width_out * kernel_h * kernel_w * (3*channel_in+2)) = static_cast<scalar_t>(0);
          
        *(col + height_out * width_out * kernel_h * kernel_w * (3*channel_in+3)) = 
            (h >= -1 && w >= -1 && h < height_in - 1 && w < width_in - 1) ? 
                im[(l_dilation_h+1) * width_in + (l_dilation_w+1)] : 
                  *(col + height_out * width_out * kernel_h * kernel_w * (3*channel_in+3)) = static_cast<scalar_t>(0);          


        col += height_out * width_out;


      }
    }
  }
}

template <typename scalar_t>
__global__ void col2im_kernel(
    const int n,
    const scalar_t* data_col,
    const scalar_t* P_h, 
    const scalar_t* P_w,  
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
    const int c_im = index / (width * height);
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
              (((c_im * kernel_h + h_k) * kernel_w + w_k) * height_col +
               h_col) *
                  width_col +
              w_col;
          val += data_col[data_col_index];
        }
      }
    }
    data_im[index] = static_cast<scalar_t>(val);
  }
}


torch::Tensor  dcls_cuda_forward(
    torch::Tensor input,    
    torch::Tensor weight,
    torch::Tensor P1,
    torch::Tensor P2,
    torch::Tensor bias,
    const int dilation_h, const int dilation_w, 
    const int stride_h, const int stride_w, 
    const int padding_h, const int padding_w, 
    const int groups) {
    
    const int batch = input.size(0);
    const int channels_in = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    const int channels_out = weight.size(0);
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);
    
    const int height_out = (height + 2 * padding_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width + 2 * padding_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    

    const int half_range_bot_h = dilation_h*kernel_h/2;
    const int half_range_top_h = half_range_bot_h - (dilation_h*kernel_h + 1)%2;

    const int half_range_bot_w = dilation_w*kernel_w/2;
    const int half_range_top_w = half_range_bot_w - (dilation_w*kernel_w +1)%2;
    
    auto P_h = at::clamp(at::floor(P1),-half_range_bot_h,half_range_top_h);
    auto rest_h = at::clamp(P1,-half_range_bot_h,half_range_top_h) - P_h;
        
    auto P_w = at::clamp(at::floor(P2),-half_range_bot_w,half_range_top_w);
    auto rest_w = at::clamp(P2,-half_range_bot_w,half_range_top_w) - P_w;
    
    P_h += dilation_h*kernel_h/2;
    P_w += dilation_w*kernel_w/2;
   
    auto ones = at::ones_like(rest_h, input.options());    
    auto W1 = (ones - rest_h) * (ones - rest_w);
    auto W2 = rest_h * (ones - rest_w);
    auto W3 = (ones - rest_h) * rest_w;
    auto W4 = rest_h * rest_w;
    auto interpolated_weight = at::empty({channels_out, channels_in/groups, 2 * kernel_h, 2 * kernel_w}, input.options());
    
    const int num_kernels_interpolation = channels_in/groups * channels_out * kernel_h * kernel_w;
    AT_DISPATCH_FLOATING_TYPES(input.type(), "dcls_forward_cuda", [&] {
        interpolation_kernel<scalar_t><<<GET_BLOCKS(num_kernels_interpolation), 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                     num_kernels_interpolation,
                                     weight.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                     W1.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     W2.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     W3.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     W4.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     channels_in/groups, channels_out,
                                     kernel_h, kernel_w, 
                                     interpolated_weight.data<scalar_t>());
    });

    // prepare group weight and bias
    auto weight_g = interpolated_weight.view({groups, channels_out/groups, channels_in/groups, 2*kernel_h, 2*kernel_w});
    auto bias_g = bias.view({groups, channels_out/groups});
    
    auto output = torch::empty({batch, channels_out , height_out , width_out}, input.options());
    const int num_kernels = channels_in * height_out * width_out;
    AT_DISPATCH_FLOATING_TYPES(input.type(), "dcls_forward_cuda", [&] {

        for (int elt = 0; elt < batch; elt++) {

            auto input_n = input.select(0, elt);
            auto output_n = output.select(0, elt);
            auto columns = at::empty({channels_in * 2 * kernel_h * 2 * kernel_w, height_out * width_out}, input.options());

            im2col_kernel<scalar_t><<<GET_BLOCKS(num_kernels), 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                             num_kernels,
                                             input_n.data<scalar_t>(),
                                             P_h.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                             P_w.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                             height, width,
                                             channels_in, channels_out,
                                             kernel_h, kernel_w, 
                                             height_out, width_out,
                                             padding_h, padding_w, 
                                             stride_h, stride_w, 
                                             dilation_h, dilation_w,
                                             groups,
                                             columns.data<scalar_t>());
            auto columns_g = columns.view({groups, channels_in/groups * 2 * kernel_h * 2 * kernel_w, height_out * width_out});
            auto output_g = output_n.view({groups, channels_out/groups, height_out * width_out});
            for (int g = 0; g < groups; ++g)
            {
                auto columns_gm = columns_g.select(0, g);
                auto weight_gm = weight_g.select(0, g).view({channels_out/groups, channels_in/groups * 2 * kernel_h * 2 * kernel_w});
                auto output_m = at::addmm(bias_g.select(0, g).view({channels_out/groups,1}),weight_gm, columns_gm);
                output_g.select(0, g) = output_m;
            }
            output.select(0, elt) = output_g.view({channels_out, height_out, width_out});
        }
    });
    
    return output;
}

std::vector<torch::Tensor> dcls_cuda_backward(
    torch::Tensor input,    
    torch::Tensor weight,
    torch::Tensor P1,
    torch::Tensor P2,
    torch::Tensor grad_output,      
    torch::Tensor bias,
    const int dilation_h, const int dilation_w, 
    const int stride_h, const int stride_w, 
    const int padding_h, const int padding_w, 
    const int groups) {
    
    auto grad_input = torch::zeros_like(input);      
    auto grad_weight = torch::zeros_like(weight);
    auto grad_P1 = torch::zeros_like(P1);
    auto grad_P2 = torch::zeros_like(P2);    
    auto grad_bias = torch::zeros_like(bias);
    
    
    const int batch = input.size(0);
    const int channels_in = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    const int channels_out = weight.size(0);
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);
    
    const int height_out = (height + 2 * padding_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width + 2 * padding_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    
    const int half_range_bot_h = dilation_h*kernel_h/2;
    const int half_range_top_h = half_range_bot_h - (dilation_h*kernel_h+1)%2;

    const int half_range_bot_w = dilation_w*kernel_w/2;
    const int half_range_top_w = half_range_bot_w - (dilation_w*kernel_w+1)%2;
    
    auto P_h = at::clamp(at::floor(P1),-half_range_bot_h,half_range_top_h);
    auto rest_h = -P_h + at::clamp(P1,-half_range_bot_h,half_range_top_h);
        
    auto P_w = at::clamp(at::floor(P2),-half_range_bot_w,half_range_top_w);
    auto rest_w = -P_w + at::clamp(P2,-half_range_bot_w,half_range_top_w);
    
    const double sigma = 0.5;
    auto ones_r = at::ones_like(rest_h, input.options());    
    auto W1 = -(ones_r - d_floor(P_h, sigma, half_range_bot_h, half_range_top_h)) * (ones_r - rest_w);
    auto W2 = -(d_floor(P_h, sigma, half_range_bot_h, half_range_top_h) - ones_r) * (ones_r - rest_w);
    auto W3 = -(ones_r - d_floor(P_h, sigma, half_range_bot_h, half_range_top_h)) * rest_w;
    auto W4 = -(d_floor(P_h, sigma, half_range_bot_h, half_range_top_h) - ones_r) * rest_w;
    
    auto interpolated_weight_P_h = at::empty({channels_out, channels_in/groups, 2 * kernel_h, 2 * kernel_w}, input.options());
    auto interpolated_weight_P_w = at::empty({channels_out, channels_in/groups, 2 * kernel_h, 2 * kernel_w}, input.options());
    
    const int num_kernels_interpolation = channels_in/groups * channels_out * kernel_h * kernel_w;
    AT_DISPATCH_FLOATING_TYPES(input.type(), "dcls_forward_cuda", [&] {
        interpolation_kernel<scalar_t><<<GET_BLOCKS(num_kernels_interpolation), 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                     num_kernels_interpolation,
                                     weight.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                     W1.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     W2.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     W3.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     W4.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     channels_in/groups, channels_out,
                                     kernel_h, kernel_w, 
                                     interpolated_weight_P_h.data<scalar_t>());
        
        W1 = -(ones_r - rest_h) * (ones_r - d_floor(P_w, sigma, half_range_bot_w, half_range_top_w));
        W2 = -rest_h * (ones_r - d_floor(P_w, sigma, half_range_bot_w, half_range_top_w));
        W3 = -(ones_r - rest_h)* (d_floor(P_w, sigma, half_range_bot_w, half_range_top_w) - ones_r);
        W4 = -rest_h * (d_floor(P_w, sigma, half_range_bot_w, half_range_top_w) - ones_r);        
        
        interpolation_kernel<scalar_t><<<GET_BLOCKS(num_kernels_interpolation), 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                     num_kernels_interpolation,
                                     weight.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                     W1.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     W2.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     W3.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     W4.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     channels_in/groups, channels_out,
                                     kernel_h, kernel_w, 
                                     interpolated_weight_P_w.data<scalar_t>());        
    });
    
    P_h += dilation_h*kernel_h/2;
    P_w += dilation_w*kernel_w/2;
    
    // prepare group weight and bias
    auto weight_g = weight.view({groups, channels_out/groups, channels_in/groups, kernel_h, kernel_w});
    auto grad_weight_g = grad_weight.view({groups, channels_out/groups, channels_in/groups, kernel_h, kernel_w});
    auto grad_bias_g = grad_bias.view({groups, channels_out/groups});
    auto ones = at::ones({height_out * width_out}, input.options());
    
    const int num_kernels = channels_in * height * width;
    const int num_kernels_im = channels_in * height_out * width_out;
    
    


    for (int elt = 0; elt < batch; elt++) {

        auto input_n = input.select(0, elt);
        auto grad_input_n = grad_input.select(0, elt);
        auto grad_output_n = grad_output.select(0, elt);   
        auto columns = at::empty({channels_in * kernel_h * kernel_w, height_out * width_out}, input.options());


        auto grad_output_g = grad_output_n.view({groups, channels_out/groups, height_out * width_out});
        auto columns_g = columns.view({groups, channels_in/groups * kernel_h * kernel_w, height_out * width_out});


        for (int g = 0; g < groups; ++g)
        {
            auto grad_output_gm = grad_output_g.select(0, g);
            auto columns_gm = columns_g.select(0, g);
            auto weight_gm = weight_g.select(0, g).view({channels_out/groups, channels_in/groups *kernel_h * kernel_w}).t();
            columns_g.select(0, g) = at::mm(weight_gm, grad_output_gm);

        }
        columns = columns_g.view({channels_in * kernel_h * kernel_w, height_out * width_out});

        AT_DISPATCH_FLOATING_TYPES(input.type(), "dcls_backward_cuda", [&] {
            col2im_kernel<scalar_t><<<GET_BLOCKS(num_kernels), 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                             num_kernels,
                                             columns.data<scalar_t>(),
                                             P_h.data<scalar_t>(),
                                             P_w.data<scalar_t>(),
                                             height, width,
                                             channels_out,
                                             kernel_h, kernel_w, 
                                             padding_h, padding_w, 
                                             stride_h, stride_w, 
                                             dilation_h, dilation_w,
                                             height_out, width_out,                
                                             grad_input_n.data<scalar_t>());

            im2col_kernel_old<scalar_t><<<GET_BLOCKS(num_kernels_im), 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                             num_kernels_im,
                                             input_n.data<scalar_t>(),
                                             P_h.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                             P_w.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                             height, width,
                                             channels_in, channels_out,               
                                             kernel_h, kernel_w, 
                                             height_out, width_out,
                                             padding_h, padding_w, 
                                             stride_h, stride_w, 
                                             dilation_h, dilation_w,
                                             groups,
                                             columns.data<scalar_t>());
        });

        for (int g = 0; g < groups; ++g)
        {
            auto grad_output_gm = grad_output_g.select(0, g);
            auto columns_gm = columns_g.select(0, g).t();
            auto grad_weight_gm = grad_weight_g.select(0, g)
                .view({channels_out/groups, channels_in/groups * kernel_h * kernel_w});
            auto grad_bias_gm = grad_bias_g.select(0, g);
            grad_weight_g.select(0, g) = at::addmm(grad_weight_gm, grad_output_gm, columns_gm)
                .view_as(grad_weight_g.select(0, g));
            grad_bias_g.select(0, g) = at::addmv(grad_bias_gm, grad_output_gm, ones);
        }
        grad_weight = grad_weight_g.view({channels_out, channels_in/groups, kernel_h, kernel_w});

        /*gather_kernel<scalar_t><<<GET_BLOCKS(num_kernels_interpolation), 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                 channels_in/groups * channels_out * kernel_h * kernel_w,
                                 grad_weight_int_g.data<scalar_t>(),
                                 channels_in/groups, channels_out,
                                 kernel_h, kernel_w, 
                                 grad_weight.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());*/

        /*gather_kernel<scalar_t><<<GET_BLOCKS(num_kernels_interpolation), 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                 channels_in/groups * channels_out * kernel_h * kernel_w,
                                 interpolated_weight_P_h.data<scalar_t>(),
                                 channels_in/groups, channels_out,
                                 kernel_h, kernel_w, 
                                 grad_Ph.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());

        gather_kernel<scalar_t><<<GET_BLOCKS(num_kernels_interpolation), 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                 channels_in/groups * channels_out * kernel_h * kernel_w,
                                 interpolated_weight_P_w.data<scalar_t>(),
                                 channels_in/groups, channels_out,
                                 kernel_h, kernel_w, 
                                 grad_Pw.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());*/ 

        auto grad_Ph = interpolated_weight_P_h.view({channels_out, channels_in/groups* kernel_h* kernel_w,4}).sum({2}); 
        auto grad_Pw = interpolated_weight_P_w.view({channels_out, channels_in/groups* kernel_h* kernel_w,4}).sum({2});

        grad_P1 += at::mm(grad_weight.view({channels_out, channels_in/groups* kernel_h* kernel_w}).t(), grad_Ph.view({channels_out, channels_in/groups * kernel_h * kernel_w})).select(1,0).view({channels_in/groups, kernel_h, kernel_w});
        grad_P2 += at::mm(grad_weight.view({channels_out, channels_in/groups* kernel_h* kernel_w}).t(), grad_Pw.view({channels_out, channels_in/groups * kernel_h * kernel_w})).select(1,0).view({channels_in/groups, kernel_h, kernel_w});
        grad_input.select(0, elt) = grad_input_n.view({channels_in, height, width});

    }


    return {grad_input,
            grad_weight,
            grad_P1,
            grad_P2,
            grad_bias};
}
