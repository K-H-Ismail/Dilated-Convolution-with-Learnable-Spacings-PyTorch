#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <math.h>
#include <vector>
#include "im2col_dcls_cuda_kernel.cu"


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
    
    P1 = P1.unsqueeze(0);
    P2 = P2.unsqueeze(0);       
    // Force batch
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
    
    // Suitable for Kaiming uniform initialization
    auto scaling_h = sqrt(kernel_h * kernel_w * channels_out * dilation_h * dilation_h)/2;
    auto scaling_w = sqrt(kernel_h * kernel_w * channels_out * dilation_w * dilation_w)/2;    
 
    const int half_range_bot_h = (dilation_h*kernel_h)/2;

    const int half_range_bot_w = (dilation_w*kernel_w)/2;
    
    auto scaled_P1 = P1*scaling_h + at::arange(-half_range_bot_h /*+ dilation_h/4*/,half_range_bot_h /*+ 1e-7*/,dilation_h, weight.options())
                            .repeat({kernel_w,1})
                            .t()
                            .repeat({1,channels_in,1,1});
    auto scaled_P2 = P2*scaling_w + at::arange(-half_range_bot_w /*+ dilation_w/4*/,half_range_bot_w /*+ 1e-7*/,dilation_w, weight.options())
                            .repeat({kernel_h,1})
                            .repeat({1,channels_in,1,1});
    
    const int limit_h = dilation_h * kernel_h;
    const int limit_w = dilation_w * kernel_w;
    
    auto P_h = scaled_P1.floor();
    auto P_w = scaled_P2.floor();    
    
    P_h += (dilation_h*kernel_h)/2 ;
    P_w += (dilation_w*kernel_w)/2 ;
    
    P_h = P_h.clamp(0,limit_h-1); 
    P_w = P_w.clamp(0,limit_w-1);    
    
    auto rest_h = (scaled_P1 + (dilation_h*kernel_h)/2).clamp(0,limit_h-1) - P_h; 
    auto rest_w = (scaled_P2 + (dilation_w*kernel_w)/2).clamp(0,limit_w-1) - P_w;    
    
    auto rhW = rest_h * weight;
    auto rwW = rest_w * weight;
    auto rhwW = rest_h * rwW;    

    // prepare group weight and bias
    auto bias_g = bias.view({groups, channels_out/groups});
    auto W1 = (weight - rhW - rwW + rhwW).view({groups, channels_out/groups, channels_in, kernel_h, kernel_w});
    auto W2 = (rhW - rhwW).view({groups, channels_out/groups, channels_in, kernel_h, kernel_w});
    auto W3 = (rwW - rhwW).view({groups, channels_out/groups, channels_in, kernel_h, kernel_w});
    auto W4 = rhwW.view({groups, channels_out/groups, channels_in, kernel_h, kernel_w}); 
  
    
    auto output = torch::empty({batch, channels_out , height_out , width_out}, input.options());
    for (int elt = 0; elt < batch; elt++) {

        auto input_n = input.select(0, elt);
        auto output_n = output.select(0, elt);

        auto input_g = input_n.view({groups, channels_in, height, width});
        auto output_g = output_n.view({groups, channels_out/groups, height_out * width_out});
        
        auto P_h_g_m = P_h.select(0, 0); 
        auto P_w_g_m = P_w.select(0, 0);        
        for (int g = 0; g < groups; ++g)
        {
            auto weights_gm = at::stack({W1.select(0, g), W3.select(0, g), W2.select(0, g), W4.select(0, g)},1);
 
            
            auto output_m =  einsum_dcls_forward_chout(input_g.select(0,g), weights_gm, P_h_g_m, P_w_g_m, dilation_h, dilation_w, padding_h, padding_w, stride_h, stride_w, height_out, width_out);
            output_g.select(0, g) = output_m;
        }
        output.select(0, elt) = output_g.view({channels_out, height_out, width_out});
    }
    
    if (!is_batch) output = output.squeeze(0);
    
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
    

    
    // Force batch
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
    
    const int half_range_bot_h = (dilation_h*kernel_h)/2;  
    const int half_range_bot_w = (dilation_w*kernel_w)/2;
    
    // Suitable for Kaiming uniform initialization
    auto scaling_h = sqrt(kernel_h * kernel_w * channels_out * dilation_h * dilation_h)/2;
    auto scaling_w = sqrt(kernel_h * kernel_w * channels_out * dilation_w * dilation_w)/2;  
    
    auto scaled_P1 = P1*scaling_h + at::arange(-half_range_bot_h /*+ dilation_h/4*/,half_range_bot_h /*+ 1e-7*/,dilation_h, weight.options())
                            .repeat({kernel_w,1})
                            .t()
                            .repeat({1,channels_in,1,1});
    auto scaled_P2 = P2*scaling_w + at::arange(-half_range_bot_w /*+ dilation_w/4*/,half_range_bot_w /*+ 1e-7*/,dilation_w, weight.options())
                            .repeat({kernel_h,1})
                            .repeat({1,channels_in,1,1});
    
    const int limit_h = dilation_h * kernel_h;
    const int limit_w = dilation_w * kernel_w;

    
    auto P_h = scaled_P1.floor();
    auto P_w = scaled_P2.floor();    
    
    P_h += (dilation_h*kernel_h)/2 ;
    P_w += (dilation_w*kernel_w)/2 ;
    
    P_h = P_h.clamp(0,limit_h-1); 
    P_w = P_w.clamp(0,limit_w-1);    
    
    auto rest_h = scaled_P1 + (dilation_h*kernel_h)/2;
    auto mask_h = rest_h.ge(0) * rest_h.le(limit_h-1);
    rest_h = rest_h.clamp(0,limit_h-1) - P_h; 
    auto rest_w = scaled_P2 + (dilation_w*kernel_w)/2;
    auto mask_w = rest_w.ge(0) * rest_w.le(limit_w-1);
    rest_w = rest_w.clamp(0,limit_w-1) - P_w;    
    

    auto rhW = rest_h * mask_w * weight;
    auto rwW = rest_w * mask_h * weight;
    auto rhw = rest_h * rest_w;   
   
    // prepare group weight and bias
    auto grad_bias_g = bias.view({groups, channels_out/groups});
    auto weight_g = weight.view({groups, channels_out/groups, channels_in, kernel_h, kernel_w});     
    auto grad_weight_g = grad_weight.view({groups, channels_out/groups, channels_in, kernel_h, kernel_w});    
    auto ones = at::ones_like(weight, weight.options()).view({groups, channels_out/groups, channels_in, kernel_h, kernel_w});
        
    auto W1 = ((ones.select(0,0) - rest_h - rest_w + rhw) * ones).view({groups, channels_out/groups, channels_in, kernel_h, kernel_w});
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
        
    
    for (int elt = 0; elt < batch; elt++) {

        auto input_n = input.select(0, elt);
        auto grad_input_n = grad_input.select(0, elt);
        auto grad_output_n = grad_output.select(0, elt);   
        auto columns = at::empty({groups * channels_in * kernel_h * kernel_w, height_out * width_out}, input.options());


        auto grad_output_g = grad_output_n.view({groups, channels_out/groups, height_out * width_out});
        auto columns_g = columns.view({groups, channels_in * kernel_h * kernel_w, height_out * width_out});

        auto input_g = input_n.view({groups, channels_in, height, width});
        
        for (int g = 0; g < groups; ++g)
        {
            auto grad_output_gm = grad_output_g.select(0, g);
            auto columns_gm = columns_g.select(0, g);
            auto weight_gm = weight_g.select(0, g).view({channels_out/groups, channels_in *kernel_h * kernel_w}).t();
            columns_g.select(0, g) = at::mm(weight_gm, grad_output_gm);

        }
        columns = columns_g.view({groups * channels_in * kernel_h * kernel_w, height_out * width_out});
        
        auto num_kernels = channels_in * kernel_h * kernel_w;
        AT_DISPATCH_FLOATING_TYPES(input.type(), "dcls_backward_cuda", [&] {
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
        });
       
        for (int g = 0; g < groups; ++g)
        {
            auto grad_output_gm = grad_output_g.select(0, g);           
            auto grad_weight_gm = grad_weight_g.select(0, g)
                .view({channels_out/groups, channels_in * kernel_h * kernel_w});           
            auto grad_bias_gm = grad_bias_g.select(0, g);
            
            auto weights_gm = at::stack({W1.select(0, g), W3.select(0, g), W2.select(0, g), W4.select(0, g)},0);
            auto weights_gm_Ph = at::stack({W1_Ph.select(0, g), W3_Ph.select(0, g), W2_Ph.select(0, g), W4_Ph.select(0, g)},0);
            auto weights_gm_Pw = at::stack({W1_Pw.select(0, g), W3_Pw.select(0, g), W2_Pw.select(0, g), W4_Pw.select(0, g)},0);            
            auto P_h_g_m = P_h.select(0,0); 
            auto P_w_g_m = P_w.select(0,0); 
            
            auto grads =  
                einsum_dcls_backward_chout(input_g.select(0,g), weights_gm,  weights_gm_Ph,  weights_gm_Pw, grad_output_gm,
                                           P_h_g_m, P_w_g_m, dilation_h, dilation_w, padding_h, padding_w, stride_h, stride_w,
                                           height_out, width_out); 
            grad_weight_g.select(0, g) = (grad_weight_gm + grads[0]).view_as(grad_weight_g.select(0, g));
            grad_P1 += grads[1].view_as(grad_P1); 
            grad_P2 += grads[2].view_as(grad_P2);            

            grad_bias_g.select(0, g) = at::addmv(grad_bias_gm, grad_output_gm, at::ones({height_out * width_out},
                                                                                        input.options()));
        }
        grad_weight = grad_weight_g.view({channels_out, channels_in, kernel_h, kernel_w});
        grad_P1 = grad_P1.view({channels_in, kernel_h, kernel_w});
        grad_P2 = grad_P2.view({channels_in, kernel_h, kernel_w});    

    }
    if (!is_batch) grad_input = grad_input.squeeze(0);

    return {grad_input,
            grad_weight,
            grad_P1*scaling_h,
            grad_P2*scaling_w,
            grad_bias};
}
