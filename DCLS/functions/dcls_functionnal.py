import torch
import im2col_dcls_2d, dcls_2d

class F_im2col_dcls2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, im, P1, P2, dilated_kernel_size, padding, stride, out_dim, shifts):
        ctx.dilated_kernel_size = dilated_kernel_size
        ctx.padding = padding
        ctx.stride = stride
        ctx.out_dim = out_dim
        ctx.shifts = shifts

        output = im2col_dcls_2d.forward(im,
                                        P1, P2,
                                        ctx.dilated_kernel_size[0], ctx.dilated_kernel_size[1],
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
    def forward(ctx, input, weight, P1, P2, bias, stride, padding, dilated_kernel_size, groups, scaling):

        ctx.stride = stride
        ctx.padding = padding
        ctx.dilated_kernel_size = dilated_kernel_size
        ctx.groups = groups
        ctx.scaling = scaling

        ctx.save_for_backward(input, weight, P1, P2, bias)

        output = dcls_2d.forward(input.float(),
                                 weight,
                                 P1,
                                 P2,
                                 bias,
                                 ctx.dilated_kernel_size[0], ctx.dilated_kernel_size[1],
                                 ctx.stride[0], ctx.stride[1],
                                 ctx.padding[0], ctx.padding[1],
                                 ctx.groups,
                                 ctx.scaling
                                )

        return output


    @staticmethod
    def backward(ctx, grad_output):
        input, weight, P1, P2, bias = ctx.saved_tensors
        outputs = dcls_2d.backward(input.float(),
                                   weight,
                                   P1,
                                   P2,
                                   grad_output.float().contiguous(),# we force grad to be float32 for now
                                   bias,
                                   ctx.dilated_kernel_size[0], ctx.dilated_kernel_size[1],
                                   ctx.stride[0], ctx.stride[1],
                                   ctx.padding[0], ctx.padding[1],
                                   ctx.groups,
                                   ctx.scaling
                                  )
        grad_input, grad_weight, grad_P1, grad_P2, grad_bias = outputs

        if bias is None:
            grad_bias = None

        return grad_input, grad_weight, grad_P1, grad_P2, grad_bias, None, None, None, None, None
