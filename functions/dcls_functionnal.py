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
        
        ctx.channels = (weight.size(0),weight.size(1))
        ctx.kernel_size = (weight.size(2),weight.size(3))
        ctx.dilation = dilation        
        ctx.stride = stride
        ctx.padding = padding
        ctx.groups = groups


        P1_range, P2_range, output_channel, input_channel = dilation[0]*ctx.kernel_size[0], dilation[1]*ctx.kernel_size[1], ctx.channels[0], ctx.channels[1]


        half_range_bot, half_range_top = (P1_range)//2 ,  (P1_range)//2-(P1_range+1)%2 
        half_range_t_bot, half_range_t_top = (P2_range)//2 ,  (P2_range)//2-(P2_range+1)%2 

        P1_ceil = P1.ceil().clamp(half_range_bot,half_range_top)
        rest1 = P1_ceil-P1.clamp(min=-half_range_bot  , max=half_range_top)
        
        P2_ceil = P2.ceil().clamp(half_range_t_bot, half_range_t_top)
        rest2 = P2_ceil-P2.clamp(min=-half_range_t_bot  , max=half_range_t_top)
                
        P1_ceil += half_range_bot
        P2_ceil += half_range_t_bot
        
        ctx.save_for_backward(input, weight, P1_ceil, rest1, P2_ceil, rest2, bias)
        
        output = dcls_cpp.forward(input, 
                                   weight, 
                                   P1_ceil, 
                                   rest1, 
                                   P2_ceil, 
                                   rest2,
                                   bias,
                                   ctx.channels[0], ctx.channels[1],
                                   ctx.kernel_size[0], ctx.kernel_size[1],
                                   ctx.dilation[0], ctx.dilation[1], 
                                   ctx.stride[0], ctx.stride[1], 
                                   ctx.padding[0], ctx.padding[1], 
                                   ctx.groups)

        return output


    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
                  
        input, weight, P1_ceil, rest1, P2_ceil, rest2, bias = ctx.saved_tensors

        outputs = dcls_cpp.backward(input, 
                                   weight, 
                                   P1_ceil, 
                                   rest1, 
                                   P2_ceil, 
                                   rest2,
                                   grad_output.contiguous(),                                    
                                   bias,
                                   ctx.channels[0], ctx.channels[1],
                                   ctx.kernel_size[0], ctx.kernel_size[1],
                                   ctx.dilation[0], ctx.dilation[1], 
                                   ctx.stride[0], ctx.stride[1], 
                                   ctx.padding[0], ctx.padding[1], 
                                   ctx.groups)
        
        grad_input, grad_weight, grad_P1, grad_P2, grad_bias = outputs
    
        return grad_input, grad_weight, grad_P1, grad_P2, grad_bias, None, None, None, None

   
