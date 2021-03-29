import torch
import numpy as np

import sys
egg_path='/home/ismail/Python-3.8.1/lib/python3.8/site-packages/dcls_full_cpp-0.0.0-py3.8-linux-x86_64.egg'
sys.path.append(egg_path)

import dcls_full_cpp


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

   
