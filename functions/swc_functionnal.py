import torch
import numpy as np

import sys
egg_path='/home/ismail/Dilated-Convolution-with-Learnable-Spacings-PyTorch/dist/sparse_weight_conv_cpp-0.0.0-py3.8-linux-x86_64.egg'
sys.path.append(egg_path)

import sparse_weight_conv_cpp


class swc2d(torch.autograd.Function):
    
    @staticmethod 
    def forward(ctx, input, weight, bias, dilation, stride, padding, groups, sparse_mm):
        
        ctx.dilation = dilation        
        ctx.stride = stride
        ctx.padding = padding
        ctx.groups = groups
        ctx.sparse_mm = sparse_mm
        
        ctx.save_for_backward(input, weight, bias)
        
        output = sparse_weight_conv_cpp.forward(input.contiguous(), 
                                                weight, 
                                                bias,
                                                ctx.dilation[0], ctx.dilation[1], 
                                                ctx.stride[0], ctx.stride[1], 
                                                ctx.padding[0], ctx.padding[1], 
                                                ctx.groups,
                                                ctx.sparse_mm)

        return output


    @staticmethod
    def backward(ctx, grad_output):
                  
        input, weight, bias = ctx.saved_tensors

        outputs = sparse_weight_conv_cpp.backward(input.contiguous(), 
                                                  weight, 
                                                  grad_output.contiguous(),                                    
                                                  bias,
                                                  ctx.dilation[0], ctx.dilation[1], 
                                                  ctx.stride[0], ctx.stride[1], 
                                                  ctx.padding[0], ctx.padding[1], 
                                                  ctx.groups,
                                                  ctx.sparse_mm)
        
        grad_input, grad_weight, grad_bias = outputs

        return grad_input, grad_weight, grad_bias, None, None, None, None, None
