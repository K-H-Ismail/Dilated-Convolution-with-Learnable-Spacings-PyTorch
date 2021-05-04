import torch
import numpy as np
from torch.autograd.function import once_differentiable

import sys
egg_path='/home/ismail/Dilated-Convolution-with-Learnable-Spacings-PyTorch/dist/dcls_full_cpp-0.0.0-py3.8-linux-x86_64.egg'
sys.path.append(egg_path)
egg_path='/home/ismail/Dilated-Convolution-with-Learnable-Spacings-PyTorch/dist/dcls_1d_cpp-0.0.0-py3.8-linux-x86_64.egg'
sys.path.append(egg_path)

import dcls_full_cpp, dcls_1d_cpp

class SurrogateDilation1d(torch.autograd.Function):
    
    @staticmethod 
    def forward(ctx, weight, P, dilation):
        
        ctx.dilation = dilation        
        
        ctx.save_for_backward(weight, P)
        
        output = dcls_1d_cpp.forward(weight,
                                       P,
                                       ctx.dilation
                                      )

        return output


    @staticmethod  
    def backward(ctx, grad_output):
                  
        weight, P = ctx.saved_tensors

        outputs = dcls_1d_cpp.backward(weight, 
                                         P, 
                                         grad_output.contiguous(),
                                         ctx.dilation
                                        )
        
        grad_weight, grad_P = outputs


        return grad_weight, grad_P, None
    
class SurrogateDilationFull(torch.autograd.Function):
    
    @staticmethod 
    def forward(ctx, weight, P1, P2, dilation):
        
        ctx.dilation = dilation        
        
        ctx.save_for_backward(weight, P1, P2)
        
        output = dcls_full_cpp.forward(weight,
                                       P1, 
                                       P2, 
                                       ctx.dilation[0], ctx.dilation[1]
                                      )

        return output


    @staticmethod   
    def backward(ctx, grad_output):
                  
        weight, P1, P2 = ctx.saved_tensors

        outputs = dcls_full_cpp.backward(weight, 
                                         P1, 
                                         P2, 
                                         grad_output.contiguous(),
                                         ctx.dilation[0], ctx.dilation[1]
                                   )
        
        grad_weight, grad_P1, grad_P2 = outputs


        return grad_weight, grad_P1, grad_P2, None

   
