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
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> P_h,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> P_w,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> W1,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> W2,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> W3,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> W4,
    const int ch_out, const int ch_in,
    const int kernel,
    const int height_out, const int width_out,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits>  interpolated_weight) {
  CUDA_KERNEL_LOOP(index, n) {
    int k = index % kernel;
    int channel_in = (index / kernel) % ch_in;
    int channel_out = (index / kernel / ch_in) % ch_out;

    int p_h = P_h[channel_out][channel_in][k];
    int p_w = P_w[channel_out][channel_in][k];
    int p_h_next = p_h + 1;
    int p_w_next = p_w + 1;

    if(p_h >= 0 & p_h < height_out & p_w >= 0 & p_w < width_out)
    {
        atomicAdd(&interpolated_weight[channel_out][channel_in][p_h][p_w], W1[channel_out][channel_in][k]);
        if(p_h_next < height_out)
            atomicAdd(&interpolated_weight[channel_out][channel_in][p_h_next][p_w], W2[channel_out][channel_in][k]);
        if(p_w_next < width_out)
            atomicAdd(&interpolated_weight[channel_out][channel_in][p_h][p_w_next], W3[channel_out][channel_in][k]);
        if(p_h_next < height_out & p_w_next < width_out)
            atomicAdd(&interpolated_weight[channel_out][channel_in][p_h_next][p_w_next], W4[channel_out][channel_in][k]);
    }

  }
}

template <typename scalar_t>
__global__ void interpolation_grad_kernel(
    const int n,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_output,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> P_h,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> P_w,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> W1,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> W2,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> W3,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> W4,
    const int ch_out, const int ch_in,
    const int kernel,
    const int height_out, const int width_out,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits>  interpolated_weight) {
  CUDA_KERNEL_LOOP(index, n) {
    int k = index % kernel;
    int channel_in = (index / kernel) % ch_in;
    int channel_out = (index / kernel / ch_in) % ch_out;

    int p_h = P_h[channel_out][channel_in][k];
    int p_w = P_w[channel_out][channel_in][k];
    int p_h_next = p_h + 1;
    int p_w_next = p_w + 1;

    if(p_h >= 0 & p_h < height_out & p_w >= 0 & p_w < width_out)
    {
        atomicAdd(&interpolated_weight[channel_out][channel_in][k],
                  grad_output[channel_out][channel_in][p_h][p_w] * W1[channel_out][channel_in][k]);

        if(p_h_next < height_out)
            atomicAdd(&interpolated_weight[channel_out][channel_in][k],
                      grad_output[channel_out][channel_in][p_h_next][p_w] * W2[channel_out][channel_in][k]);

        if(p_w_next < width_out)
            atomicAdd(&interpolated_weight[channel_out][channel_in][k],
                      grad_output[channel_out][channel_in][p_h][p_w_next] * W3[channel_out][channel_in][k]);

        if(p_h_next < height_out & p_w_next < width_out)
            atomicAdd(&interpolated_weight[channel_out][channel_in][k],
                      grad_output[channel_out][channel_in][p_h_next][p_w_next] * W4[channel_out][channel_in][k]);
    }
  }
}

torch::Tensor  dcls_construct_2d_cuda_forward(
    torch::Tensor weight,
    torch::Tensor P1,
    torch::Tensor P2,
    const int dilated_kernel_size_h, const int dilated_kernel_size_w,
    const float scaling
    ) {

    const int channels_out = weight.size(0);
    const int channels_in = weight.size(1);
    const int kernel = weight.size(2);


    // Bounds for Ph and Pw
    const int half_range_h = dilated_kernel_size_h / 2;
    const int half_range_w = dilated_kernel_size_w / 2;

    // Preform scaling
    auto scaled_P1 = P1 * scaling /*+ at::arange(-half_range_h, half_range_h, dilation_h, weight.options())
                                      .repeat({kernel_w,1})
                                      .t()
                                      .repeat({channels_out,channels_in,1,1})
                                    + ((kernel_h - 1) * dilation_h / 4)*/;
    auto scaled_P2 = P2 * scaling /*+ at::arange(-half_range_w, half_range_w, dilation_w, weight.options())
                                      .repeat({kernel_h,1})
                                      .repeat({channels_out,channels_in,1,1})
                                    + ((kernel_w - 1) * dilation_w / 4)*/;

    // Add d.k/2, positions are now normal around d.k/2
    auto P_h = scaled_P1 + half_range_h;
    auto P_w = scaled_P2 + half_range_w;

    // Apply floor function, positions are now integers
    P_h = P_h.floor();
    P_w = P_w.floor();

    // Apply clamp function, positions are now integers strictly between 0 and d.k - 1
    P_h = P_h.clamp(0, dilated_kernel_size_h - 1);
    P_w = P_w.clamp(0, dilated_kernel_size_w - 1);

    // Calculate rests for interpolation
    auto rest_h = (scaled_P1 + half_range_h).clamp(0, dilated_kernel_size_h - 1) - P_h;
    auto rest_w = (scaled_P2 + half_range_w).clamp(0, dilated_kernel_size_w - 1) - P_w;

    auto rhW = rest_h * weight;
    auto rwW = rest_w * weight;
    auto rhwW = rest_h * rwW;
    auto W1 = weight - rhW - rwW + rhwW;
    auto W2 = rhW - rhwW;
    auto W3 = rwW - rhwW;
    auto W4 = rhwW;

    auto output = torch::zeros({channels_out, channels_in, dilated_kernel_size_h, dilated_kernel_size_w}, weight.options());

    const int num_kernels =  channels_out * channels_in * kernel;
    AT_DISPATCH_FLOATING_TYPES(weight.type(), "dcls_construct_2d_forward_cuda", [&] {

        interpolation_kernel<scalar_t><<<GET_BLOCKS(num_kernels), 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                     num_kernels,
                                     P_h.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     P_w.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     W1.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     W2.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     W3.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     W4.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     channels_out, channels_in,
                                     kernel,
                                     dilated_kernel_size_h, dilated_kernel_size_w,
                                     output.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
    });
    return output;
}

std::vector<torch::Tensor> dcls_construct_2d_cuda_backward(
    torch::Tensor weight,
    torch::Tensor P1,
    torch::Tensor P2,
    torch::Tensor grad_output,
    const int dilated_kernel_size_h, const int dilated_kernel_size_w,
    const float scaling
    ) {

    auto grad_weight = torch::zeros_like(weight);
    auto grad_P1 = torch::zeros_like(P1);
    auto grad_P2 = torch::zeros_like(P2);

    const int channels_out = weight.size(0);
    const int channels_in = weight.size(1);
    const int kernel = weight.size(2);


    // Bounds for Ph and Pw
    const int half_range_h = dilated_kernel_size_h / 2;
    const int half_range_w = dilated_kernel_size_w / 2;

    // Preform scaling
    auto scaled_P1 = P1 * scaling /*+ at::arange(-half_range_h, half_range_h, dilation_h, weight.options())
                                      .repeat({kernel_w,1})
                                      .t()
                                      .repeat({channels_out,channels_in,1,1})
                                    + ((kernel_h - 1) * dilation_h / 4)*/;
    auto scaled_P2 = P2 * scaling /*+ at::arange(-half_range_w, half_range_w, dilation_w, weight.options())
                                      .repeat({kernel_h,1})
                                      .repeat({channels_out,channels_in,1,1})
                                    + ((kernel_w - 1) * dilation_w / 4)*/;


    // Add d.k/2, positions are now normal around d.k/2
    auto P_h = scaled_P1 + half_range_h;
    auto P_w = scaled_P2 + half_range_w;

    // Apply floor function, positions are now integers
    P_h = P_h.floor();
    P_w = P_w.floor();

    // Apply clamp function, positions are now integers strictly between 0 and d.k - 1
    P_h = P_h.clamp(0, dilated_kernel_size_h - 1);
    P_w = P_w.clamp(0, dilated_kernel_size_w - 1);

    // Calculate rests for interpolation
    auto rest_h = scaled_P1 + half_range_h;
    auto mask_h = rest_h.gt(0) * rest_h.lt(dilated_kernel_size_h-1);
    rest_h = rest_h.clamp(0,dilated_kernel_size_h-1) - P_h;
    auto rest_w = scaled_P2 + half_range_w;
    auto mask_w = rest_w.gt(0) * rest_w.lt(dilated_kernel_size_w-1);
    rest_w = rest_w.clamp(0,dilated_kernel_size_w-1) - P_w;


    auto rhW = rest_h * weight * mask_h;
    auto rwW = rest_w * weight * mask_w;
    auto rhw = rest_h * rest_w;

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


    const int num_kernels = channels_out * channels_in * kernel;
    AT_DISPATCH_FLOATING_TYPES(weight.type(), "dcls_construct_2d_backward_cuda", [&] {

        interpolation_grad_kernel<scalar_t><<<GET_BLOCKS(num_kernels), 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                     num_kernels,
                                     grad_output.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                     P_h.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     P_w.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     W1.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     W2.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     W3.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     W4.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     channels_out, channels_in,
                                     kernel,
                                     dilated_kernel_size_h, dilated_kernel_size_w,
                                     grad_weight.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
        interpolation_grad_kernel<scalar_t><<<GET_BLOCKS(num_kernels), 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                     num_kernels,
                                     grad_output.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                     P_h.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     P_w.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     W1_Ph.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     W2_Ph.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     W3_Ph.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     W4_Ph.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     channels_out, channels_in,
                                     kernel,
                                     dilated_kernel_size_h, dilated_kernel_size_w,
                                     grad_P1.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
        interpolation_grad_kernel<scalar_t><<<GET_BLOCKS(num_kernels), 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                     num_kernels,
                                     grad_output.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                     P_h.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     P_w.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     W1_Pw.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     W2_Pw.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     W3_Pw.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     W4_Pw.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     channels_out, channels_in,
                                     kernel,
                                     dilated_kernel_size_h, dilated_kernel_size_w,
                                     grad_P2.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());

    });


    return {grad_weight,
            grad_P1*scaling,
            grad_P2*scaling};
}
