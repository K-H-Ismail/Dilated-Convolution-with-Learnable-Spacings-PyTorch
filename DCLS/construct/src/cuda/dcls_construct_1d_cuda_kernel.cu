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

    int p = P[channel_out][channel_in][l_out];
    int p_next = p + 1;

    if(p >= 0 & p < length_out)
    {
        atomicAdd(&interpolated_weight[channel_out][channel_in][p], W1[channel_out][channel_in][l_out]);
        if(p_next < length_out)
            atomicAdd(&interpolated_weight[channel_out][channel_in][p_next], W2[channel_out][channel_in][l_out]);
    }
  }
}

template <typename scalar_t>
__global__ void interpolation_grad_kernel(
    const int n,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_output,
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

    int p = P[channel_out][channel_in][l_out];
    int p_next = p + 1;

    if(p >= 0 & p < length_out)
    {
        atomicAdd(&interpolated_weight[channel_out][channel_in][l_out],
                  grad_output[channel_out][channel_in][p] * W1[channel_out][channel_in][l_out]);

        if(p_next < length_out)
            atomicAdd(&interpolated_weight[channel_out][channel_in][l_out],
                      grad_output[channel_out][channel_in][p_next] * W2[channel_out][channel_in][l_out]);
    }

  }
}

torch::Tensor  dcls_construct_1d_cuda_forward(
    torch::Tensor weight,
    torch::Tensor P1,
    const int dilated_kernel_size,
    const float scaling
    ) {

    const int channels_out = weight.size(0);
    const int channels_in = weight.size(1);
    const int kernel = weight.size(2);

    // Bound for P
    const int half_range = dilated_kernel_size / 2;

    // Preform scaling
    auto scaled_P = P1 * scaling /*+ at::arange(-half_range, half_range, dilation, weight.options())
                                      .repeat({channels_out,channels_in,1})
                                    + ((kernel - 1) * dilation / 4)*/;

    // Add d.k/2, positions are now normal around d.k/2
    auto P = scaled_P + half_range;

    // Apply floor function, positions are now integers
    P = P.floor();

    // Apply clamp function, positions are now integers strictly between 0 and d.k - 1
    P = P.clamp(0, dilated_kernel_size - 1);

    // Calculate rests for interpolation
    auto rest = (scaled_P + half_range).clamp(0, dilated_kernel_size - 1) - P;

    auto W2 = rest * weight;
    auto W1 = weight - W2;

    auto output = torch::zeros({channels_out, channels_in, dilated_kernel_size}, weight.options());

    const int num_kernels =  channels_out * channels_in * kernel;
    AT_DISPATCH_FLOATING_TYPES(weight.type(), "dcls_construct_1d_forward_cuda", [&] {

        interpolation_kernel<scalar_t><<<GET_BLOCKS(num_kernels), 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                     num_kernels,
                                     P.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     W1.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     W2.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     channels_out, channels_in,
                                     kernel,
                                     dilated_kernel_size,
                                     output.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
    });
    return output;
}

std::vector<torch::Tensor> dcls_construct_1d_cuda_backward(
    torch::Tensor weight,
    torch::Tensor P1,
    torch::Tensor grad_output,
    const int dilated_kernel_size,
    const float scaling
    ) {

    auto grad_weight = torch::zeros_like(weight);
    auto grad_P1 = torch::zeros_like(P1);

    const int channels_out = weight.size(0);
    const int channels_in = weight.size(1);
    const int kernel = weight.size(2);

    // Bound for P
    const int half_range = dilated_kernel_size / 2;

    // Preform scaling
    auto scaled_P = P1 * scaling /*+ at::arange(-half_range, half_range, dilation, weight.options())
                                      .repeat({channels_out,channels_in,1})
                                    + ((kernel - 1) * dilation / 4)*/;

    // Add d.k/2, positions are now normal around d.k/2
    auto P = scaled_P + half_range;

    // Apply floor function, positions are now integers
    P = P.floor();

    // Apply clamp function, positions are now integers strictly between 0 and d.k - 1
    P = P.clamp(0, dilated_kernel_size - 1);

    // Calculate rests for interpolation

    auto rest = scaled_P + half_range;
    auto mask = rest.ge(0) * rest.le(dilated_kernel_size-1);
    rest = rest.clamp(0,dilated_kernel_size-1) - P;

    auto ones = at::ones_like(rest, weight.options());
    auto W2 = rest;
    auto W1 = ones - W2;

    auto W1_P = - weight * mask;
    auto W2_P = - W1_P;


    const int num_kernels = channels_out * channels_in * kernel;
    AT_DISPATCH_FLOATING_TYPES(weight.type(), "dcls_construct_1d_backward_cuda", [&] {

        interpolation_grad_kernel<scalar_t><<<GET_BLOCKS(num_kernels), 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                     num_kernels,
                                     grad_output.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     P.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     W1.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     W2.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     channels_out, channels_in,
                                     kernel,
                                     dilated_kernel_size,
                                     grad_weight.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
        interpolation_grad_kernel<scalar_t><<<GET_BLOCKS(num_kernels), 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                     num_kernels,
                                     grad_output.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     P.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     W1_P.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     W2_P.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                                     channels_out, channels_in,
                                     kernel,
                                     dilated_kernel_size,
                                     grad_P1.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());

    });

    return {grad_weight,
            grad_P1*scaling};
}
