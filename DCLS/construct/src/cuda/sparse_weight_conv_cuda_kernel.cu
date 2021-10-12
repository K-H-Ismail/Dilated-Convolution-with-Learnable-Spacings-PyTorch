#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/div_rtn.h>
#include <ATen/cuda/CUDABlas.h>
#include "spmm_cuda_kernel.cu"

static torch::Tensor newViewWeightMM2d(torch::Tensor weight) {
  if (weight.dim() == 4) {
    int64_t s1 = weight.size(0);
    int64_t s2 = weight.size(1) * weight.size(2) * weight.size(3);
    weight = weight.view({s1, s2});
  }
  return weight;
}

void SpatialConvolutionMM_updateOutput(
           torch::Tensor input,
           torch::Tensor output,
           torch::Tensor weight,
           bool is_bias,           
           torch::Tensor bias,
           torch::Tensor columns,
           int kW, int kH,
           int dW, int dH,
           int padW, int padH,
           int strW, int strH) {

  input = input.contiguous();
  output = output.contiguous(); 
  weight = weight.contiguous();
  bias = bias.contiguous();
  columns = columns.contiguous(); 
    
  weight = newViewWeightMM2d(weight);

  int ndim = input.dim();
  int dimf = 0;
  int dimh = 1;
  int dimw = 2;

  if (ndim == 4) {
    dimf++;
    dimh++;
    dimw++;
  }

  int64_t nInputPlane = input.size(dimf);
  int64_t inputHeight  = input.size(dimh);
  int64_t inputWidth   = input.size(dimw);
  int64_t nOutputPlane = weight.size(0);
  int64_t outputHeight = (inputHeight + 2*padH - (dH * (kH - 1) + 1)) / strH + 1;
  int64_t outputWidth  = (inputWidth + 2*padW - (dW * (kW - 1) + 1)) / strW + 1;


  int is_batch = 1;
  if (input.dim() == 3) {
    // Force batch
    is_batch = 0;
    input = input.unsqueeze(0);
  }

  // Batch size + input planes
  int64_t batchSize = input.size(0);

  // Resize output
  output = output.view({batchSize, nOutputPlane, outputHeight, outputWidth});

  // Resize temporary columns
  columns = columns.view({nInputPlane*kW*kH, outputHeight*outputWidth});

  // Helpers
  auto input_n = torch::empty({nInputPlane, inputHeight , inputWidth}, input.options());
  auto output_n = torch::empty({nOutputPlane, outputHeight , outputWidth}, input.options());

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per output:
    input_n = input.select(0, elt);
    output_n = output.select(0, elt);

    // Extract columns:
    columns = at::im2col(input_n, {kH,kW}, {dH,dW}, {padH,padW}, {strH,strW});
    
    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    if (is_bias) {
        sparse_mm_dense_cusparse_backend(columns.get_device(), weight.size(0), weight.size(1), columns.size(1), weight.data_ptr<float>(), columns.data_ptr<float>(), output_n.data_ptr<float>());
        output_n = output_n.add(bias);
    } else {
      sparse_mm_dense_cusparse_backend(columns.get_device(), weight.size(0), weight.size(1), columns.size(1), weight.data_ptr<float>(), columns.data_ptr<float>(), output_n.data_ptr<float>());
    }    
          

  }

  // Free
  //THCTensor_(free)(state, input_n);
  //THCTensor_(free)(state, output_n);

  // Resize output
  if (is_batch == 0) {
    output = output.view({nOutputPlane, outputHeight, outputWidth});
    input = input.view({nInputPlane, inputHeight, inputWidth});
  }

  //THCTensor_(free)(state, input);
  //THCTensor_(free)(state, weight);
}

void SpatialConvolutionMM_updateGradInput(
           torch::Tensor input,
           torch::Tensor gradOutput,
           torch::Tensor gradInput,
           torch::Tensor weight,
           torch::Tensor gradColumns,
           torch::Tensor ones,
           int kW, int kH,
           int dW, int dH,
           int padW, int padH,
           int strW, int strH) {
    
  input = input.contiguous();
  gradOutput = gradOutput.contiguous(); 
  gradInput = gradInput.contiguous();
  weight = weight.contiguous();
  gradColumns = gradColumns.contiguous();
  ones = ones.contiguous(); 
    
  weight = newViewWeightMM2d(weight);

  // Params
  int nInputPlane = weight.dim() == 2 ? weight.size(1)/(kW*kH) : weight.size(1);
  int nOutputPlane = weight.size(0);


  int is_batch = 1;
  if (input.dim() == 3) {
    // Force batch
    is_batch = 0;
    input = input.unsqueeze(0);
    gradOutput = gradOutput.unsqueeze(0);
  }

  int64_t inputWidth   = input.size(3);
  int64_t inputHeight  = input.size(2);
  int64_t outputHeight = (inputHeight + 2*padH - (dH * (kH - 1) + 1)) / strH + 1;
  int64_t outputWidth  = (inputWidth + 2*padW - (dW * (kW - 1) + 1)) / strW + 1;

  // Batch size + input planes
  int64_t batchSize = input.size(0);

  // Resize output
  gradInput = gradInput.view({batchSize, nInputPlane, inputHeight, inputWidth});

  // Resize temporary columns
  gradColumns = gradColumns.view({nInputPlane*kW*kH, outputHeight*outputWidth});

  // Helpers
  auto gradInput_n = torch::empty({nInputPlane, inputHeight , inputWidth}, input.options());
  auto gradOutput_n = torch::empty({nOutputPlane, outputHeight , outputWidth}, input.options());    

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per output:
    gradInput_n = gradInput.select(0, elt);
    gradOutput_n = gradOutput.select(0, elt);

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    //gradColumns = at::mm(weight.t(), gradOutput_n.view({nOutputPlane, outputHeight * outputWidth})); 
    sparse_mm_dense_cusparse_backend(gradOutput_n.get_device(), weight.size(1), weight.size(0), outputHeight * outputWidth, weight.t().data_ptr<float>(), gradOutput_n.view({nOutputPlane, outputHeight * outputWidth}).data_ptr<float>(), gradColumns.data_ptr<float>());
    // Unpack columns back into input:
    gradInput_n = at::col2im(gradColumns, {inputHeight,inputWidth}, {kH,kW}, {dH,dW}, {padH,padW}, {strH,strW});
    gradInput.select(0, elt) = gradInput_n.squeeze(0);

  }

  // Free
  //THCTensor_(free)(state, gradInput_n);
  //THCTensor_(free)(state, gradOutput_n);
  //THCTensor_(free)(state, weight);

  // Resize output
  if (is_batch == 0) {
    gradOutput = gradOutput.view({nOutputPlane, outputHeight, outputWidth});
    input = input.view({nInputPlane, inputHeight, inputWidth});
    gradInput = gradInput.view({nInputPlane, inputHeight, inputWidth});
  }

  //THCTensor_(free)(state, input);
  //THCTensor_(free)(state, gradOutput);
}

void SpatialConvolutionMM_accGradParameters(
           torch::Tensor input,
           torch::Tensor gradOutput,
           bool is_gradweight,
           torch::Tensor gradWeight,
           bool is_gradbias,    
           torch::Tensor gradBias,
           torch::Tensor columns,
           torch::Tensor ones,
           int kW, int kH,
           int dW, int dH,
           int padW, int padH,
           int strW, int strH) {
    
  input = input.contiguous();
  gradOutput = gradOutput.contiguous(); 
  gradWeight = gradWeight.contiguous();
  gradBias = gradBias.contiguous();
  columns = columns.contiguous();
  ones = ones.contiguous(); 
    
  int is_batch = 1;
  if (input.dim() == 3) {
    // Force batch
    is_batch = 0;
    input = input.unsqueeze(0);
    gradOutput = gradOutput.unsqueeze(0);
  }

  int64_t nInputPlane = input.size(1);
  int64_t nOutputPlane = gradOutput.size(1);

  int64_t inputWidth   = input.size(3);
  int64_t inputHeight  = input.size(2);
  int64_t outputHeight = (inputHeight + 2*padH - (dH * (kH - 1) + 1)) / strH + 1;
  int64_t outputWidth  = (inputWidth + 2*padW - (dW * (kW - 1) + 1)) / strW + 1;

  // Batch size + input planes
  int64_t batchSize = input.size(0);

  // Define a buffer of ones, for bias accumulation
  if (ones.dim() != 2 || ones.size(0)*ones.size(1) < outputHeight*outputWidth) {
    // Resize plane and fill with ones...
    ones.view({outputHeight, outputWidth});
    ones.fill_(1);
  }

  // Resize temporary columns
  columns = columns.view({nInputPlane*kW*kH, outputHeight*outputWidth});
  
  // Helpers
  auto input_n = torch::empty({nInputPlane, inputHeight , inputWidth}, input.options());
  auto gradOutput_n = torch::empty({nOutputPlane, outputHeight , outputWidth}, input.options());
  
  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per output:
    gradOutput_n = gradOutput.select(0, elt);
      
    // Do Weight:
    if (is_gradweight) {
      // Matrix mulitply per output:
      input_n = input.select(0, elt);
      
      // Extract columns:
      columns = at::im2col(input_n, {kH,kW}, {dH,dW}, {padH,padW}, {strH,strW});

      // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
      gradWeight += at::mm(gradOutput_n.view({nOutputPlane, outputHeight * outputWidth}), columns.t())
                          .view_as({gradWeight});
    }

    // Do Bias:
    if (is_gradbias) {
        
      // Do GEMV (note: this is a bit confusing because gemv assumes column-major matrices)
      gradBias += at::mv(gradOutput_n.view({nOutputPlane, outputHeight * outputWidth}), ones.view({outputHeight * outputWidth}));
    }
  }

  // Free
  //THCTensor_(free)(state, input_n);
  //THCTensor_(free)(state, gradOutput_n);
  //if (gradWeight)
  //  THCTensor_(free)(state, gradWeight);

  // Resize
  if (is_batch == 0) {
    gradOutput = gradOutput.view({nOutputPlane, outputHeight, outputWidth});
    input = input.view({nInputPlane, inputHeight, inputWidth});
  }

  //THCTensor_(free)(state, input);
  //THCTensor_(free)(state, gradOutput);
}


torch::Tensor  sparse_weight_conv_cuda_forward(  
    torch::Tensor input,    
    torch::Tensor weight,
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
    
    auto output = torch::empty({batch, channels_out , height_out , width_out}, input.options());
    auto columns = at::empty({channels_in * kernel_h * kernel_w, height_out * width_out}, input.options());   
    
    SpatialConvolutionMM_updateOutput(
           input,
           output,
           weight,
           false,
           bias,
           columns,
           kernel_w, kernel_h,
           dilation_w, dilation_h,
           padding_w, padding_h,
           stride_w, stride_h);
    
    return output;
}


std::vector<torch::Tensor> sparse_weight_conv_cuda_backward(   
    torch::Tensor input,    
    torch::Tensor weight,
    torch::Tensor grad_output,      
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
    
    auto columns = at::empty({channels_in * kernel_h * kernel_w, height_out * width_out}, input.options());
    auto ones = torch::ones({height_out, width_out}, input.options());    

    auto grad_input = torch::zeros_like(input);      
    auto grad_weight = torch::zeros_like(weight);
    auto grad_bias = torch::zeros_like(bias);
    
    SpatialConvolutionMM_updateGradInput(
           input,
           grad_output,
           grad_input,
           weight,
           columns,
           ones,
           kernel_w, kernel_h,
           dilation_w, dilation_h,
           padding_w, padding_h,
           stride_w, stride_h);
    
    SpatialConvolutionMM_accGradParameters(
           input,
           grad_output,
           true,
           grad_weight,
           false,
           grad_bias,
           columns,
           ones,
           kernel_w, kernel_h,
           dilation_w, dilation_h,
           padding_w, padding_h,
           stride_w, stride_h);    
    
    return {grad_input,
            grad_weight,
            grad_bias};
}
