import torch
import numpy as np
from torch.autograd.function import once_differentiable

import sys
egg_path='/home/ismail/Python-3.8.1/lib/python3.8/site-packages/dcls_cpp-0.0.0-py3.8-linux-x86_64.egg'
sys.path.append(egg_path)

import dcls_cpp


class SurrogateDilation(torch.autograd.Function):
    
    @staticmethod 
    def forward(ctx, input, weight, P1, P2, bias, dilation, stride, padding, groups):
        
        ctx.kernel_size = (weight.size(2),weight.size(3))
        ctx.dilation = dilation        
        ctx.stride = stride
        ctx.padding = padding
        ctx.groups = groups
        
        ctx.save_for_backward(input, weight, P1, P2, bias)
        
        output = dcls_cpp.forward(input.contiguous(), 
                                   weight, 
                                   P1, 
                                   P2, 
                                   bias,
                                   ctx.dilation[0], ctx.dilation[1], 
                                   ctx.stride[0], ctx.stride[1], 
                                   ctx.padding[0], ctx.padding[1], 
                                   ctx.groups)

        return output


    @staticmethod
    def backward(ctx, grad_output):
                  
        input, weight, P1, P2, bias = ctx.saved_tensors

        outputs = dcls_cpp.backward(input.contiguous(), 
                                   weight, 
                                   P1, 
                                   P2, 
                                   grad_output.contiguous(),                                    
                                   bias,
                                   ctx.dilation[0], ctx.dilation[1], 
                                   ctx.stride[0], ctx.stride[1], 
                                   ctx.padding[0], ctx.padding[1], 
                                   ctx.groups)
        
        grad_input, grad_weight, grad_P1, grad_P2, grad_bias = outputs
        #print(grad_weight)
        #print(grad_P1)
        #print(grad_P2)

        return grad_input, grad_weight, grad_P1, grad_P2, grad_bias, None, None, None, None

   
