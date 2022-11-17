import torch
import numpy as np
from torch.autograd.function import once_differentiable
import torch.nn.functional as F
import sys
import os
global eps
eps = torch.finfo(torch.float32).eps
import dcls_construct_1d, dcls_construct_2d, dcls_construct_3d#, dcls_construct_2_1d, dcls_construct_3_1d, dcls_construct_3_2d

class ConstructKernel(torch.autograd.Function):
    pass

class ConstructKernel1d(ConstructKernel):

    @staticmethod
    def forward(ctx, weight, P, dilated_kernel_size, scaling):

        ctx.dilated_kernel_size = dilated_kernel_size
        ctx.scaling = scaling

        ctx.save_for_backward(weight, P)

        output = dcls_construct_1d.forward(weight,
                                           P,
                                           ctx.dilated_kernel_size[0],
                                           ctx.scaling
                                          )

        return output

    @staticmethod
    def backward(ctx, grad_output):

        weight, P = ctx.saved_tensors

        outputs = dcls_construct_1d.backward(weight,
                                             P,
                                             grad_output.contiguous(),
                                             ctx.dilated_kernel_size[0],
                                             ctx.scaling
                                            )

        grad_weight, grad_P = outputs

        return grad_weight, grad_P, None, None

class ConstructKernel2d(ConstructKernel):

    @staticmethod
    def forward(ctx, weight, P1, P2, dilated_kernel_size, scaling):

        ctx.dilated_kernel_size = dilated_kernel_size
        ctx.scaling = scaling

        ctx.save_for_backward(weight, P1.contiguous(), P2.contiguous())

        output = dcls_construct_2d.forward(weight,
                                           P1.contiguous(),
                                           P2.contiguous(),
                                           ctx.dilated_kernel_size[0],
                                           ctx.dilated_kernel_size[1],
                                           ctx.scaling
                                          )
        return output


    @staticmethod
    def backward(ctx, grad_output):

        weight, P1, P2 = ctx.saved_tensors

        outputs = dcls_construct_2d.backward(weight,
                                             P1,
                                             P2,
                                             grad_output.contiguous(),
                                             ctx.dilated_kernel_size[0],
                                             ctx.dilated_kernel_size[1],
                                             ctx.scaling
                                            )

        grad_weight, grad_P1, grad_P2 = outputs

        return grad_weight, grad_P1, grad_P2, None, None

class ConstructKernel3d(ConstructKernel):

    @staticmethod
    def forward(ctx, weight, P1, P2, P3, dilated_kernel_size, scaling):

        ctx.dilated_kernel_size = dilated_kernel_size
        ctx.scaling = scaling

        ctx.save_for_backward(weight, P1, P2, P3)

        output = dcls_construct_3d.forward(weight,
                                       P1,
                                       P2,
                                       P3,
                                       ctx.dilated_kernel_size[0],
                                       ctx.dilated_kernel_size[1],
                                       ctx.dilated_kernel_size[2],
                                       ctx.scaling
                                      )

        return output


    @staticmethod
    def backward(ctx, grad_output):

        weight, P1, P2, P3 = ctx.saved_tensors

        outputs = dcls_construct_3d.backward(weight,
                                         P1,
                                         P2,
                                         P3,
                                         grad_output.contiguous(),
                                         ctx.dilated_kernel_size[0],
                                         ctx.dilated_kernel_size[1],
                                         ctx.dilated_kernel_size[2],
                                         ctx.scaling
                                   )

        grad_weight, grad_P1, grad_P2, grad_P3 = outputs


        return grad_weight, grad_P1, grad_P2, grad_P3, None, None

class ConstructKernel2_1d(ConstructKernel):

    @staticmethod
    def forward(ctx, weight, P, dilation):

        ctx.dilation = dilation

        ctx.save_for_backward(weight, P)

        output = dcls_construct_2_1d.forward(weight,
                                       P,
                                       ctx.dilation[0]
                                      )
        return output

    @staticmethod
    def backward(ctx, grad_output):

        weight, P = ctx.saved_tensors

        outputs = dcls_construct_2_1d.backward(weight,
                                         P,
                                         grad_output.contiguous(),
                                         ctx.dilation[0]
                                        )

        grad_weight, grad_P = outputs

        return grad_weight, grad_P, None

class ConstructKernel3_1d(ConstructKernel):

    @staticmethod
    def forward(ctx, weight, P, dilation, gain):

        ctx.dilation = dilation
        ctx.gain

        ctx.save_for_backward(weight, P)

        output = dcls_construct_3_1d.forward(weight,
                                       P,
                                       ctx.dilation[0],
                                       ctx.gain
                                      )
        return output

    @staticmethod
    def backward(ctx, grad_output):

        weight, P = ctx.saved_tensors

        outputs = dcls_construct_3_1d.backward(weight,
                                         P,
                                         grad_output.contiguous(),
                                         ctx.dilation[0],
                                         ctx.gain
                                        )

        grad_weight, grad_P = outputs

        return grad_weight, grad_P, None

class ConstructKernel3_2d(ConstructKernel):

    @staticmethod
    def forward(ctx, weight, P1, P2, dilation):

        ctx.dilation = dilation

        ctx.save_for_backward(weight, P1, P2)

        output = dcls_construct_3_2d.forward(weight,
                                       P1,
                                       P2,
                                       ctx.dilation[0], ctx.dilation[1]
                                      )

        return output


    @staticmethod
    def backward(ctx, grad_output):

        weight, P1, P2 = ctx.saved_tensors

        outputs = dcls_construct_3_2d.backward(weight,
                                         P1,
                                         P2,
                                         grad_output.contiguous(),
                                         ctx.dilation[0], ctx.dilation[1]
                                   )

        grad_weight, grad_P1, grad_P2 = outputs


        return grad_weight, grad_P1, grad_P2, None


