import torch
import numpy as np
from torch.autograd.function import once_differentiable
import torch.nn.functional as F
import sys
import os

import sparse_weight_conv


class swc2d(torch.autograd.Function):
    
    @staticmethod 
    def forward(ctx, input, weight, bias, stride, padding, dilation, groups):
        
        ctx.dilation = dilation        
        ctx.stride = stride
        ctx.padding = padding
        ctx.groups = groups
        
        ctx.save_for_backward(input, weight, bias)
        
        output = sparse_weight_conv.forward(input.contiguous(), 
                                                weight, 
                                                bias,
                                                ctx.dilation[0], ctx.dilation[1], 
                                                ctx.stride[0], ctx.stride[1], 
                                                ctx.padding[0], ctx.padding[1], 
                                                ctx.groups)

        return output


    @staticmethod
    def backward(ctx, grad_output):
                  
        input, weight, bias = ctx.saved_tensors

        outputs = sparse_weight_conv.backward(input.contiguous(), 
                                                  weight, 
                                                  grad_output.contiguous(),                                    
                                                  bias,
                                                  ctx.dilation[0], ctx.dilation[1], 
                                                  ctx.stride[0], ctx.stride[1], 
                                                  ctx.padding[0], ctx.padding[1], 
                                                  ctx.groups)
        
        grad_input, grad_weight, grad_bias = outputs

        return grad_input, grad_weight, grad_bias, None, None, None, None
