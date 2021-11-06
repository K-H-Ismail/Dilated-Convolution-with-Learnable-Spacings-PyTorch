import torch
import im2col_dcls_2d, dcls_2d

class F_im2col_dcls2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, im, P1, P2, dilation, padding, stride, out_dim, shifts):
        ctx.dilation = dilation 
        ctx.padding = padding 
        ctx.stride = stride
        ctx.out_dim = out_dim
        ctx.shifts = shifts         
        
        output = im2col_dcls_2d.forward(im,
                                        P1, P2,
                                        ctx.dilation[0], ctx.dilation[1],
                                        ctx.padding[0], ctx.padding[1],
                                        ctx.stride[0], ctx.stride[1],
                                        ctx.out_dim[0], ctx.out_dim[1],                                     
                                        ctx.shifts[0], ctx.shifts[1]
                                        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None, None, None, None, None, None
    

class dcls2d_conv(torch.autograd.Function):

    @staticmethod 
    def forward(ctx, input, weight, P1, P2, bias, stride, padding, dilation, groups, sign_grad, gain):
        
        ctx.stride = stride 
        ctx.padding = padding 
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.sign_grad = sign_grad        
        ctx.gain = gain
        
        ctx.save_for_backward(input, weight, P1, P2, bias)
        
        output = dcls_2d.forward(input,
                                 weight, 
                                 P1, 
                                 P2, 
                                 bias,
                                 ctx.dilation[0], ctx.dilation[1],
                                 ctx.stride[0], ctx.stride[1],
                                 ctx.padding[0], ctx.padding[1],
                                 ctx.groups,
                                 ctx.gain
                                )

        return output


    @staticmethod   
    def backward(ctx, grad_output):        
        input, weight, P1, P2, bias = ctx.saved_tensors
        outputs = dcls_2d.backward(input,
                                   weight, 
                                   P1, 
                                   P2, 
                                   grad_output.float().contiguous(),# we force grad to be float32 for now
                                   bias,
                                   ctx.dilation[0], ctx.dilation[1],
                                   ctx.stride[0], ctx.stride[1],
                                   ctx.padding[0], ctx.padding[1],
                                   ctx.groups,
                                   ctx.gain 
                                  )
        if ctx.sign_grad :
            grad_P1 = grad_P1.sign()
            grad_P2 = grad_P2.sign()            
        
        grad_input, grad_weight, grad_P1, grad_P2, grad_bias = outputs
        return grad_input, grad_weight, grad_P1, grad_P2, grad_bias, None, None, None, None, None, None   