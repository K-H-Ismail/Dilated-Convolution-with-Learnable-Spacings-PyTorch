import torch
import torch.nn as nn
import torch.nn.functional as F

import spmm

class sparse_mm_dense_atf(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sparse: torch.Tensor, dense: torch.Tensor):
        # sparse: [M, N]  dense: [N, P]  y:[M, P]
        if sparse.requires_grad or dense.requires_grad:
            ctx.save_for_backward(sparse, dense)
        y = torch.zeros(size=[sparse.shape[0], dense.shape[1]], dtype=torch.float, device=sparse.device)
        spmm.sparse_mm_dense_cusparse(sparse, dense, y)
        # y = torch.mm(sparse, dense)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: [M, P]
        sparse, dense = ctx.saved_tensors
        grad_sparse = grad_dense = None
        if ctx.needs_input_grad[0]:
            grad_sparse = grad_output.mm(dense.t())
        if ctx.needs_input_grad[1]:
            grad_dense = torch.zeros_like(dense.data)
            spmm.sparse_mm_dense_cusparse(sparse.t(), grad_output, grad_dense)
            # grad_dense = sparse.t().mm(grad_output)
        return grad_sparse, grad_dense


def sparse_mm_dense(sparse: torch.Tensor, dense: torch.Tensor):
    return sparse_mm_dense_atf.apply(sparse, dense)




