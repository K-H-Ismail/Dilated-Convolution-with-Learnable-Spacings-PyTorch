#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <math.h>
#include <vector>
#include "im2col_dcls_2d_cuda_kernel.cu"

// Forward method for dcls 2d with no kernel construction
torch::Tensor  dcls_2d_cuda_forward(
    torch::Tensor input,    
    torch::Tensor weight,
    torch::Tensor P1,
    torch::Tensor P2,
    torch::Tensor bias,
    const int dilation_h, const int dilation_w, 
    const int stride_h, const int stride_w, 
    const int padding_h, const int padding_w, 
    const int groups,
    const float gain) {
    
    // Unsqueeze P1 and P2 for element-wise matrix multiplication compatibility
    P1 = P1.unsqueeze(0);
    P2 = P2.unsqueeze(0);
    
    // Force batch if input is of dim 3
    auto is_batch = true;
    if (input.dim() == 3) {
        is_batch = false;
        input = input.unsqueeze(0);
    }
        
    const int batch = input.size(0);
    const int channels_in = weight.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    const int channels_out = weight.size(0);
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);
    
    const int height_out = (height + 2 * padding_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width + 2 * padding_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    
    // Suitable scaling for Kaiming uniform initialization
    auto scaling_h = gain * sqrt(kernel_h * kernel_w * channels_in);
    auto scaling_w = gain * sqrt(kernel_h * kernel_w * channels_in);     
     
    // Bounds for Ph and Pw
    const int half_range_h = (dilation_h * kernel_h) / 2;
    const int half_range_w = (dilation_w * kernel_w) / 2;
    
    // Preform scaling and add regular spacings
    auto scaled_P1 = P1 * scaling_h + at::arange(-half_range_h, half_range_h, dilation_h, weight.options())
                                      .repeat({kernel_w,1})
                                      .t()
                                      .repeat({1,channels_in,1,1})
                                    + ((kernel_h - 1) * dilation_h / 4);
    auto scaled_P2 = P2 * scaling_w + at::arange(-half_range_w, half_range_w, dilation_w, weight.options())
                                      .repeat({kernel_h,1})
                                      .repeat({1,channels_in,1,1})
                                    + ((kernel_w - 1) * dilation_w / 4);
    
    // Limits of the dilated kernel
    const int limit_h = dilation_h * kernel_h;
    const int limit_w = dilation_w * kernel_w;
    
    // Add d.k/2, positions are now uniformly around 0 and d.k - 1    
    auto P_h = scaled_P1 + half_range_h;
    auto P_w = scaled_P2 + half_range_w;    
    
    // Apply floor function, positions are now integers uniformly around 0 and d.k - 1
    P_h = P_h.floor();
    P_w = P_w.floor();
    
    // Apply clamp function, positions are now integers strictly between 0 and d.k - 1
    P_h = P_h.clamp(0, limit_h - 1); 
    P_w = P_w.clamp(0, limit_w - 1);    
    
    // Calculate rests for interpolation
    auto rest_h = (scaled_P1 + half_range_h).clamp(0, limit_h - 1) - P_h; 
    auto rest_w = (scaled_P2 + half_range_w).clamp(0, limit_w - 1) - P_w;    
    
    // Calculate interpolations and make groups for separable conv    
    auto rhW = rest_h * weight;
    auto rwW = rest_w * weight;
    auto rhwW = rest_h * rwW;    
    
    auto bias_g = bias.view({groups, channels_out/groups});
    auto W1 = (weight - rhW - rwW + rhwW).view({groups, channels_out/groups, channels_in, kernel_h, kernel_w});
    auto W2 = (rhW - rhwW).view({groups, channels_out/groups, channels_in, kernel_h, kernel_w});
    auto W3 = (rwW - rhwW).view({groups, channels_out/groups, channels_in, kernel_h, kernel_w});
    auto W4 = rhwW.view({groups, channels_out/groups, channels_in, kernel_h, kernel_w}); 

    // We consider the maximum free memory 
    long free_memory = constants::GLOBAL_FREE_MEMORY;
    
    // Choose chunksize according to total memory (we consider 2d interpolation and float32 tensors thus 4 x 4)
    const int max_chunk_size = free_memory / (4 * 4 * channels_in * groups * kernel_h * kernel_w * height_out * width_out) + 1;
    
    const int nb_chunks = (batch - 1) / max_chunk_size + 1;
    
    auto chunked_input = input.chunk(nb_chunks,0);
    
    auto output = at::zeros({}, input.options());
    auto P_h_g_m = P_h.select(0, 0); 
    auto P_w_g_m = P_w.select(0, 0);    
    auto weights_gm = at::stack({W1, 
                                 W2, 
                                 W3, 
                                 W4},2);    
    // Loop over batch chunks
    for (int chunk = 0; chunk < nb_chunks; chunk++) {

        auto input_n = chunked_input[chunk];
        const int chunk_size = input_n.size(0);

        auto input_g = input_n.view({chunk_size, groups, channels_in, height, width});       
        auto output_g = at::zeros({chunk_size, groups, channels_out/groups, height_out * width_out}, input.options());
        
        // Call im2col_dcls + matmul
        auto output_m =  mm_dcls_2d_forward(input_g, weights_gm, P_h_g_m, P_w_g_m, 
                                         dilation_h, dilation_w, padding_h, padding_w, 
                                         stride_h, stride_w, height_out, width_out);
            
        auto output_chunk = output_m.view({chunk_size, channels_out, height_out, width_out});
        
        // Concatenate outputs along chunks
        output = chunk == 0 ?  output_chunk : at::cat({output, output_chunk},0);
    }
    
    // Only if input was of dim 3
    if (!is_batch) output = output.squeeze(0);

    return output;
}

// Backward method for dcls 2d with no kernel construction
std::vector<torch::Tensor> dcls_2d_cuda_backward(
    torch::Tensor input,    
    torch::Tensor weight,
    torch::Tensor P1,
    torch::Tensor P2,
    torch::Tensor grad_output,      
    torch::Tensor bias,
    const int dilation_h, const int dilation_w, 
    const int stride_h, const int stride_w, 
    const int padding_h, const int padding_w, 
    const int groups,
    const float gain) {
        
    // Force batch if input is of dim 3
    auto is_batch = true;
    if (input.dim() == 3) {
        is_batch = false;
        input = input.unsqueeze(0);
    }
    
    auto grad_input = torch::zeros_like(input);      
    auto grad_weight = torch::zeros_like(weight);
    auto grad_P1 = torch::zeros_like(P1);
    auto grad_P2 = torch::zeros_like(P2);    
    auto grad_bias = torch::zeros_like(bias);
       
    // Unsqueeze P1 and P2 for element-wise matrix multiplication compatibility    
    P1 = P1.unsqueeze(0);
    P2 = P2.unsqueeze(0); 
    
    const int batch = input.size(0);
    const int channels_in = weight.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    const int channels_out = weight.size(0);
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);
    
    const int height_out = (height + 2 * padding_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width + 2 * padding_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    
    // Suitable scaling for Kaiming uniform initialization
    auto scaling_h = gain * sqrt(kernel_h * kernel_w * channels_in);
    auto scaling_w = gain * sqrt(kernel_h * kernel_w * channels_in);    

    // Bounds for Ph and Pw
    const int half_range_h = (dilation_h * kernel_h) / 2;
    const int half_range_w = (dilation_w * kernel_w) / 2;

    // Preform scaling and add regular spacings
    auto scaled_P1 = P1 * scaling_h + at::arange(-half_range_h, half_range_h, dilation_h, weight.options())
                                      .repeat({kernel_w,1})
                                      .t()
                                      .repeat({1,channels_in,1,1})
                                    + ((kernel_h - 1) * dilation_h / 4);
    auto scaled_P2 = P2 * scaling_w + at::arange(-half_range_w, half_range_w, dilation_w, weight.options())
                                      .repeat({kernel_h,1})
                                      .repeat({1,channels_in,1,1})
                                    + ((kernel_w - 1) * dilation_w / 4);
    
    // Limits of the dilated kernel
    const int limit_h = dilation_h * kernel_h;
    const int limit_w = dilation_w * kernel_w;
    
    // Add d.k/2, positions are now uniformly around 0 and d.k - 1    
    auto P_h = scaled_P1 + half_range_h;
    auto P_w = scaled_P2 + half_range_w;    
    
    // Apply floor function, positions are now integers uniformly around 0 and d.k - 1
    P_h = P_h.floor();
    P_w = P_w.floor();

    // Apply clamp function, positions are now integers strictly between 0 and d.k - 1
    P_h = P_h.clamp(0, limit_h - 1); 
    P_w = P_w.clamp(0, limit_w - 1);     
    
    // Calculate rests and masks for interpolation
    auto rest_h = scaled_P1 + half_range_h;
    auto mask_h = rest_h.ge(0) * rest_h.le(limit_h - 1);
    rest_h = rest_h.clamp(0, limit_h - 1) - P_h; 
    auto rest_w = scaled_P2 + half_range_w;
    auto mask_w = rest_w.ge(0) * rest_w.le(limit_w - 1);
    rest_w = rest_w.clamp(0,limit_w - 1) - P_w;

    auto rhW = rest_h * mask_w * weight;
    auto rwW = rest_w * mask_h * weight;
    auto rhw = rest_h * rest_w;   
   
    // Calculate interpolations and make groups for separable conv 
    auto grad_bias_g = bias.view({groups, channels_out/groups});
    auto weight_g = weight.view({groups, channels_out/groups, channels_in, kernel_h, kernel_w});     
    auto grad_weight_g = grad_weight.view({groups, channels_out/groups, channels_in, kernel_h, kernel_w});    
    auto ones = at::ones_like(weight, weight.options()).view({groups, channels_out/groups, channels_in, kernel_h, kernel_w});
        
    auto W1 = ((ones.select(0,0) - rest_h - rest_w + rhw) * ones)
              .view({groups, channels_out/groups, channels_in, kernel_h, kernel_w});
    auto W2 = ((rest_h - rhw) * ones).view({groups, channels_out/groups, channels_in, kernel_h, kernel_w});
    auto W3 = ((rest_w - rhw) * ones).view({groups, channels_out/groups, channels_in, kernel_h, kernel_w});
    auto W4 = (rhw * ones).view({groups, channels_out/groups, channels_in, kernel_h, kernel_w}); 
    
    auto W1_Ph = (-weight * mask_h + rwW).view({groups, channels_out/groups, channels_in, kernel_h, kernel_w});
    auto W2_Ph = -W1_Ph.view({groups, channels_out/groups, channels_in, kernel_h, kernel_w});
    auto W3_Ph = -rwW.view({groups, channels_out/groups, channels_in, kernel_h, kernel_w});
    auto W4_Ph = -W3_Ph.view({groups, channels_out/groups, channels_in, kernel_h, kernel_w});
    
    auto W1_Pw = (-weight * mask_w + rhW).view({groups, channels_out/groups, channels_in, kernel_h, kernel_w});
    auto W2_Pw = -rhW.view({groups, channels_out/groups, channels_in, kernel_h, kernel_w});
    auto W3_Pw = -W1_Pw.view({groups, channels_out/groups, channels_in, kernel_h, kernel_w});
    auto W4_Pw = -W2_Pw.view({groups, channels_out/groups, channels_in, kernel_h, kernel_w});
    
    // We consider the maximum free memory 
    long free_memory = constants::GLOBAL_FREE_MEMORY;    
    
    // Choose chunksize according to total memory (we consider 2d interpolation and float32 tensors thus 4 x 4)
    const int max_chunk_size = free_memory / (4 * 4 * channels_in * groups * kernel_h * kernel_w * height_out * width_out) + 1;
    const int nb_chunks = (batch - 1) / max_chunk_size + 1;
    
    auto chunked_input = input.chunk(nb_chunks,0);
    auto chunked_grad_input = grad_input.chunk(nb_chunks,0);
    auto chunked_grad_output = grad_output.chunk(nb_chunks,0);    

    auto P_h_g_m = P_h.select(0, 0); 
    auto P_w_g_m = P_w.select(0, 0);
    
    auto weights_gm = at::stack({W1, 
                                 W2, 
                                 W3, 
                                 W4},1);
    auto weights_gm_Ph = at::stack({W1_Ph, 
                                    W2_Ph, 
                                    W3_Ph, 
                                    W4_Ph},1);
    auto weights_gm_Pw = at::stack({W1_Pw, 
                                    W2_Pw, 
                                    W3_Pw, 
                                    W4_Pw},1);
    auto weight_gm = weight_g.view({groups, channels_out/groups, channels_in * kernel_h * kernel_w}).permute({0,2,1});

    // Loop over batch chunks    
    for (int chunk = 0; chunk < nb_chunks; chunk++) {

        auto input_n = chunked_input[chunk];
        const int chunk_size = input_n.size(0);
        
        auto grad_input_n = chunked_grad_input[chunk];
        auto grad_output_n = chunked_grad_output[chunk];   
        auto columns = at::empty({chunk_size, groups * channels_in * kernel_h * kernel_w, height_out * width_out}, input.options());

        auto grad_output_g = grad_output_n.view({chunk_size, groups, channels_out/groups, height_out * width_out});
        auto columns_g = columns.view({chunk_size, groups, channels_in * kernel_h * kernel_w, height_out * width_out});
        auto input_g = input_n.view({chunk_size, groups, channels_in, height, width});
        
        // Col2im for the gradient with respect to the input
       /* columns_g = at::matmul(weight_gm, grad_output_g);
        
        columns = columns_g.view({chunk_size, groups * channels_in * kernel_h * kernel_w, height_out * width_out});
        
        auto num_kernels = chunk_size * channels_in * groups * height * width;
        AT_DISPATCH_FLOATING_TYPES(input.type(), "col2im_dcls_2d_backward_cuda", [&] {
            col2im_kernel<scalar_t><<<GET_BLOCKS(num_kernels), 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
                                             num_kernels,
                                             columns.data<scalar_t>(),
                                             height, width,
                                             channels_out,
                                             kernel_h, kernel_w, 
                                             padding_h, padding_w, 
                                             stride_h, stride_w, 
                                             dilation_h, dilation_w,
                                             height_out, width_out,                
                                             grad_input_n.data<scalar_t>());
        });*/
      
            
        // Call im2col_dcls + matmul
        auto grads =  mm_dcls_2d_backward(input_g, weights_gm,  
                                       weights_gm_Ph,  weights_gm_Pw, grad_output_g, 
                                       P_h_g_m, P_w_g_m, dilation_h, dilation_w, 
                                       padding_h, padding_w, stride_h, stride_w,
                                       height_out, width_out);

        grad_weight += grads[0].view({channels_out, channels_in, kernel_h, kernel_w});
        grad_P1 += grads[1].view_as(grad_P1); 
        grad_P2 += grads[2].view_as(grad_P2);

        // Batch-matrix times vector multiplication is applied to calculate the gradient of the bias,
        // then we sum over chunk size
        grad_bias +=  at::matmul(grad_output_g, at::ones({height_out * width_out}, weight.options())).sum(0).view_as(grad_bias);


    }
                                    
    // Only if input was of dim 3    
    if (!is_batch) grad_input = grad_input.squeeze(0);

    return {grad_input,
            grad_weight,
            grad_P1 * scaling_h, // apply the scaling
            grad_P2 * scaling_w, // apply the scaling
            grad_bias};
}
