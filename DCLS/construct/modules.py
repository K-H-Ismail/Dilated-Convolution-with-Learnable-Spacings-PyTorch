# coding=utf-8
import math
import warnings

import torch
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn import init
from torch.nn.modules import Module
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Optional, List, Tuple
import logging
global install_implicit_gemm
try:
    from depthwise_conv2d_implicit_gemm import _DepthWiseConv2dImplicitGEMMFP32, _DepthWiseConv2dImplicitGEMMFP16
    install_implicit_gemm = True
except ImportError as error:
    install_implicit_gemm = False
except Exception as exception:
    install_implicit_gemm = False

convolution_notes = \
    {"groups_note": r"""* :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,
        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters (of size
          :math:`\frac{\text{out\_channels}}{\text{in\_channels}}`).""",

        "depthwise_separable_note": r"""When `groups == in_channels` and `out_channels == K * in_channels`,
        where `K` is a positive integer, this operation is also known as a "depthwise convolution".
        In other words, for an input of size :math:`(N, C_{in}, L_{in})`,
        a depthwise convolution with a depthwise multiplier `K` can be performed with the arguments
        :math:`(C_\text{in}=C_\text{in}, C_\text{out}=C_\text{in} \times \text{K}, ..., \text{groups}=C_\text{in})`."""}  # noqa: B950

class _DclsNd(Module):

    __constants__ = ['stride', 'padding', 'dilated_kernel_size', 'groups',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_count']
    __annotations__ = {'bias': Optional[torch.Tensor]}

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
        ...

    _in_channels: int
    out_channels: int
    kernel_count: int
    stride: Tuple[int, ...]
    padding: Tuple[int, ...]
    dilated_kernel_size: Tuple[int, ...]
    transposed: bool
    output_padding: Tuple[int, ...]
    groups: int
    padding_mode: str
    weight: Tensor
    bias: Optional[Tensor]

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_count: int,
                 stride: Tuple[int, ...],
                 padding: Tuple[int, ...],
                 dilated_kernel_size: Tuple[int, ...],
                 transposed: bool,
                 output_padding: Tuple[int, ...],
                 groups: int,
                 bias: bool,
                 padding_mode: str) -> None:
        super(_DclsNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_count = kernel_count
        self.stride = stride
        self.padding = padding
        self.dilated_kernel_size = dilated_kernel_size
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, kernel_count))
            self.P = Parameter(torch.Tensor(len(dilated_kernel_size), in_channels // groups,
                                            out_channels, kernel_count))

        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, kernel_count))
            self.P = Parameter(torch.Tensor(len(dilated_kernel_size),
                                            out_channels, in_channels // groups, kernel_count))
        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        for i in range(len(self.dilated_kernel_size)):
            lim = self.dilated_kernel_size[i] // 2
            with torch.no_grad():
                init.normal_(self.P.select(0,i), 0, 0.5).clamp(-lim, lim)

    def clamp_parameters(self) -> None:
        for i in range(len(self.dilated_kernel_size)):
            with torch.no_grad():
                lim = self.dilated_kernel_size[i] // 2
                self.P.select(0,i).clamp_(-lim, lim)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_count={kernel_count} (previous kernel_size)'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilated_kernel_size != (1,) * len(self.dilated_kernel_size):
            s += ', dilated_kernel_size={dilated_kernel_size} (learnable)'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(_DclsNd, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'

class ConstructKernel1d(Module):
    def __init__(self, out_channels, in_channels, groups, kernel_count, dilated_kernel_size):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.groups = groups
        self.dilated_kernel_size = dilated_kernel_size
        self.kernel_count = kernel_count
        I = torch.arange(0, dilated_kernel_size[0])
        I = I.expand(out_channels, in_channels//groups, kernel_count,-1).permute(3,0,1,2)
        self.I = Parameter(I, requires_grad=False)

        self.lim = torch.zeros(1)
        self.lim[0] = dilated_kernel_size[0]
        self.lim = self.lim.expand(out_channels, in_channels//groups, 
        			   kernel_count, -1).permute(3,0,1,2)
        self.lim = Parameter(self.lim, requires_grad=False)
        
    def forward(self, W, P):        
        P = P + self.lim // 2
        Pr = P
        P = P.floor()
        R = (Pr - P).expand(self.dilated_kernel_size[0],-1,-1,-1,-1)
        R1 = R.select(2,0); P1 = P.select(0,0)       
        cond1 = (self.I == P1)
        cond2 = (self.I == P1+1)
        W1 = torch.where(cond1, 1.0, 0.0)
        W2 = torch.where(cond2, 1.0, 0.0)

        K = W1 + R1 * (W2 - W1)
        K = W * K 
        K = K.sum(3) 
        K = K.permute(1,2,0)
        return K

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_count={kernel_count}')
        if self.dilated_kernel_size != (1,) * len(self.dilated_kernel_size):
            s += ', dilated_kernel_size={dilated_kernel_size}'
        if self.groups != 1:
            s += ', groups={groups}'
        return s.format(**self.__dict__)
        
class ConstructKernel2d(Module):
    def __init__(self, out_channels, in_channels, groups, kernel_count, dilated_kernel_size):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.groups = groups
        self.dilated_kernel_size = dilated_kernel_size
        self.kernel_count = kernel_count
        J = torch.arange(0, dilated_kernel_size[0]).expand(dilated_kernel_size[1],-1)
        I = torch.arange(0, dilated_kernel_size[1]).expand(dilated_kernel_size[0],-1)
        I = I.expand(out_channels, in_channels//groups, kernel_count,-1,-1).permute(3,4,0,1,2)
        J = J.expand(out_channels, in_channels//groups, kernel_count,-1,-1).permute(4,3,0,1,2)

        self.I = Parameter(I, requires_grad=False)
        self.J = Parameter(J, requires_grad=False)
        self.lim = torch.zeros(2)
        self.lim[0] = dilated_kernel_size[0]; self.lim[1] = dilated_kernel_size[1];
        self.lim = self.lim.expand(out_channels, in_channels//groups, 
        			   kernel_count, -1).permute(3,0,1,2)
        self.lim = Parameter(self.lim, requires_grad=False)
        
    def forward(self, W, P):        
        P = P + self.lim // 2
        Pr = P
        P = P.floor()
        R = (Pr - P).expand(self.dilated_kernel_size[0], self.dilated_kernel_size[1],-1,-1,-1,-1)
        R1 = R.select(2,0); P1 = P.select(0,0)
        R2 = R.select(2,1); P2 = P.select(0,1)
        R1R2 = R1*R2        
        cond1 = (self.I == P1)
        cond2 = (self.J == P2)
        cond3 = (self.I == P1+1)
        cond4 = (self.J == P2+1)       
        W1 = torch.where(cond1*cond2, 1.0, 0.0)
        W2 = torch.where(cond1*cond4, 1.0, 0.0)
        W3 = torch.where(cond3*cond2, 1.0, 0.0) 
        W4 = torch.where(cond3*cond4, 1.0, 0.0)
        K = W1 + R1R2*(W1 - W2 - W3 + W4) + R1*(W3 - W1) + R2*(W2-W1)
        K = W * K 
        K = K.sum(4) 
        K = K.permute(2,3,0,1)
        return K

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_count={kernel_count}')
        if self.dilated_kernel_size != (1,) * len(self.dilated_kernel_size):
            s += ', dilated_kernel_size={dilated_kernel_size}'
        if self.groups != 1:
            s += ', groups={groups}'
        return s.format(**self.__dict__)

class ConstructKernel3d(Module):
    def __init__(self, out_channels, in_channels, groups, kernel_count, dilated_kernel_size):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.groups = groups
        self.dilated_kernel_size = dilated_kernel_size
        self.kernel_count = kernel_count
        L = torch.arange(0,dilated_kernel_size[0]).expand(dilated_kernel_size[1],dilated_kernel_size[2],-1)
        J = torch.arange(0,dilated_kernel_size[1]).expand(dilated_kernel_size[0],dilated_kernel_size[2],-1)
        I = torch.arange(0,dilated_kernel_size[2]).expand(dilated_kernel_size[0],dilated_kernel_size[1],-1)
        L = L.expand(out_channels,in_channels//groups,kernel_count,-1,-1,-1).permute(5,3,4,0,1,2)
        I = I.expand(out_channels,in_channels//groups,kernel_count,-1,-1,-1).permute(3,4,5,0,1,2)
        J = J.expand(out_channels,in_channels//groups,kernel_count,-1,-1,-1).permute(3,5,4,0,1,2)
        self.L = Parameter(L, requires_grad=False)
        self.I = Parameter(I, requires_grad=False)
        self.J = Parameter(J, requires_grad=False)
        self.lim = torch.zeros(3)
        self.lim[0] = dilated_kernel_size[0] 
        self.lim[1] = dilated_kernel_size[1]
        self.lim[2] = dilated_kernel_size[2]
        self.lim = self.lim.expand(out_channels, in_channels//groups, 
        			   kernel_count, -1).permute(3,0,1,2)
        self.lim = Parameter(self.lim, requires_grad=False)
        
    def forward(self, W, P):        
        P = P + self.lim // 2
        Pr = P
        P = P.floor()
        R = (Pr - P).expand(self.dilated_kernel_size[0], self.dilated_kernel_size[1], self.dilated_kernel_size[2],-1,-1,-1,-1)
        R1 = R.select(3,0); P1 = P.select(0,0)
        R2 = R.select(3,1); P2 = P.select(0,1)
        R3 = R.select(3,2); P3 = P.select(0,2)
        #R1R2 = R1*R2    
        #R1R3 = R1*R2
        #R2R3 = R2*R3
        #R1R2R3 = R1R2*R3    
        cond1 = (self.L == P1)
        cond2 = (self.I == P2)
        cond3 = (self.J == P3)
        cond4 = (self.L == P1+1)
        cond5 = (self.I == P2+1)
        cond6 = (self.J == P3+1)     
        W1 = torch.where(cond1*cond2*cond3, 1.0, 0.0)
        W2 = torch.where(cond4*cond2*cond3, 1.0, 0.0)
        W3 = torch.where(cond1*cond5*cond3, 1.0, 0.0) 
        W4 = torch.where(cond4*cond5*cond3, 1.0, 0.0)
        W5 = torch.where(cond1*cond2*cond6, 1.0, 0.0)
        W6 = torch.where(cond4*cond2*cond6, 1.0, 0.0)
        W7 = torch.where(cond1*cond5*cond6, 1.0, 0.0) 
        W8 = torch.where(cond4*cond5*cond6, 1.0, 0.0)
        # needs a better computing
        K  = W1 * (1 - R1) * (1 - R2) * (1 - R3)
        K += W2 * R1 	  * (1 - R2) * (1 - R3)
        K += W3 * (1 - R1) * R2 	     * (1 - R3)
        K += W4 * R1 	  * R2       * (1 - R3)
        K += W5 * (1 - R1) * (1 - R2) * R3
        K += W6 * R1 	  * (1 - R2) * R3
        K += W7 * (1 - R1) * R2 	     * R3
        K += W8 * R1	  * R2 	     * R3                                
        K = W * K 
        K = K.sum(5) 
        K = K.permute(3,4,0,1,2)
        return K

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_count={kernel_count}')
        if self.dilated_kernel_size != (1,) * len(self.dilated_kernel_size):
            s += ', dilated_kernel_size={dilated_kernel_size}'
        if self.groups != 1:
            s += ', groups={groups}'
        return s.format(**self.__dict__)
class Dcls1d(_DclsNd):
    __doc__ = r"""

    Shape:
        - Input: :math:`(N, C_{in}, L_{in})` or :math:`(C_{in}, L_{in})`
        - Output: :math:`(N, C_{out}, L_{out})` or :math:`(C_{out}, L_{out})`, where

          .. math::
              L_{out} = \left\lfloor\frac{L_{in} + 2 \times \text{padding} - \text{dilation}
                        \times (\text{kernel\_size} - 1) - 1}{\text{stride}} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
            :math:`(\text{out\_channels},
            \frac{\text{in\_channels}}{\text{groups}}, \text{kernel\_size})`.
            The values of these weights are sampled from
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{groups}{C_\text{in} * \text{kernel\_size}}`
        bias (Tensor):   the learnable bias of the module of shape
            (out_channels). If :attr:`bias` is ``True``, then the values of these weights are
            sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{groups}{C_\text{in} * \text{kernel\_size}}`

    Examples::

        >>> m = nn.Conv1d(16, 33, 3, stride=2)
        >>> input = torch.randn(20, 16, 50)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_count: int,
        stride: _size_1_t = 1,
        padding: _size_1_t = 0,
        dilated_kernel_size: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
    ):
        stride_ = _single(stride)
        padding_ = _single(padding)
        dilated_kernel_size_ = _single(dilated_kernel_size)
        super(Dcls1d, self).__init__(
            in_channels, out_channels, kernel_count, stride_, padding_, dilated_kernel_size_,
            False, _single(0), groups, bias, padding_mode)

        self.DCK = ConstructKernel1d(self.out_channels, 
        			     self.in_channels, 
        			     self.groups, 
        			     self.kernel_count, 
        			     self.dilated_kernel_size)
    def extra_repr(self):
        s = super(Dcls1d, self).extra_repr()
        return s.format(**self.__dict__)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor], P: Tensor):
            if self.padding_mode != 'zeros':
                return F.conv1d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                self.DCK(weight, P), bias,
                                self.stride, _single(0), _single(1), self.groups)
            return F.conv1d(input, self.DCK(weight, P), bias,
                                   self.stride, self.padding, _single(1), self.groups)

    def forward(self, input: Tensor) -> Tensor:
            return self._conv_forward(input, self.weight, self.bias, self.P)

class Dcls2d(_DclsNd):
    __doc__ = r"""Applies a 2D convolution over an input signal composed of several input
    planes.
    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{\text{in}}, H, W)` and output :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})`
    can be precisely described as:
    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)
    where :math:`\star` is the valid 2D `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`H` is a height of input planes in pixels, and :math:`W` is
    width in pixels.
    """ + r"""
    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.
    * :attr:`stride` controls the stride for the cross-correlation, a single
      number or a tuple.
    * :attr:`padding` controls the amount of implicit padding on both
      sides for :attr:`padding` number of points for each dimension.
    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.
    {groups_note}
    The parameters :attr:`kernel_count`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:
        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension
    Note:
        {depthwise_separable_note}
    Note:
        {cudnn_reproducibility_note}
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_count (int): Number of elements in the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0
        padding_mode (string, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        dilated_kernel_size (int or tuple, optional): Size of dilated kernel. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``
    """ + r"""
    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where
          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                        \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor
          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                        \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor
    Attributes:
        weight (Tensor): the learnable weights of the module of shape
            :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}},`
            :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]})`.
            The values of these weights are sampled from
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
        bias (Tensor):   the learnable bias of the module of shape
            (out_channels). If :attr:`bias` is ``True``,
            then the values of these weights are
            sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
    Examples:
        >>> # With square kernels and equal stride
        >>> m = nn.Conv2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> # non-square kernels and unequal stride and with padding and dilation
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
        >>> input = torch.randn(20, 16, 50, 100)
        >>> output = m(input)
    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation
    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_count: int,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilated_kernel_size: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        use_implicit_gemm: bool = True
    ):
        stride_ = _pair(stride)
        padding_ = _pair(padding)
        dilated_kernel_size_ = _pair(dilated_kernel_size)
        super(Dcls2d, self).__init__(
            in_channels, out_channels, kernel_count, stride_, padding_, dilated_kernel_size_,
            False, _pair(0), groups, bias, padding_mode)

        self.cond = (self.in_channels == self.out_channels == self.groups
                and self.padding[0] ==  self.dilated_kernel_size[0] // 2
                and self.padding[1] ==  self.dilated_kernel_size[1] // 2)
        if not torch.cuda.is_available():
            logging.warning('DepthWiseConv2dImplicitGEMM requires cuda, switching to native conv2d')
        if (not install_implicit_gemm) and use_implicit_gemm:
            logging.warning('DepthWiseConv2dImplicitGEMM not installed')
        if (not self.cond) and use_implicit_gemm:
            logging.warning('to use DepthWiseConv2dImplicitGEMM you must have: \n \
                            (in_channels == out_channels == groups) and (padding == dilated_kernel_size // 2)')
        if (not install_implicit_gemm) or (not self.cond):
            logging.warning('switching to native conv2d')
        self.use_implicit_gemm = use_implicit_gemm and install_implicit_gemm and self.cond and torch.cuda.is_available()

        self.DCK = ConstructKernel2d(self.out_channels, 
        			     self.in_channels, 
        			     self.groups, 
        			     self.kernel_count, 
        			     self.dilated_kernel_size)
    def extra_repr(self):
        s = super(Dcls2d, self).extra_repr()
        if self.use_implicit_gemm:
            s += ', (using DepthWiseConv2dImplicitGEMM)'
        else:
            s += ', (using torch im2col GEMM)'
        return s.format(**self.__dict__)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor], P: Tensor):
        if self.use_implicit_gemm:
            if input.dtype == torch.float32:
                x = _DepthWiseConv2dImplicitGEMMFP32.apply(
                    input, self.DCK(weight, P).contiguous())
            elif input.dtype == torch.float16:
                x = _DepthWiseConv2dImplicitGEMMFP16(
                    input, self.DCK(weight, P).contiguous())
            else:
                raise TypeError("Only support fp32 and fp16, get {}".format(x.dtype))
            if self.bias is not None:
                x = x + self.bias.to(x).view(1, -1, 1, 1)
            return x
        else:
            if self.padding_mode != 'zeros':
                return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                self.DCK(weight, P), bias,
                                self.stride, _pair(0), _pair(1), self.groups)
            return F.conv2d(input, self.DCK(weight, P), bias,
                                   self.stride, self.padding, _pair(1), self.groups)

    def forward(self, input: Tensor) -> Tensor:
            return self._conv_forward(input, self.weight, self.bias, self.P);



class Dcls3d(_DclsNd):
    __doc__ = r"""

    Shape:
        - Input: :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})` or :math:`(C_{in}, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` or :math:`(C_{out}, D_{out}, H_{out}, W_{out})`,
          where

          .. math::
              D_{out} = \left\lfloor\frac{D_{in} + 2 \times \text{padding}[0] - \text{dilation}[0]
                    \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              H_{out} = \left\lfloor\frac{H_{in} + 2 \times \text{padding}[1] - \text{dilation}[1]
                    \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in} + 2 \times \text{padding}[2] - \text{dilation}[2]
                    \times (\text{kernel\_size}[2] - 1) - 1}{\text{stride}[2]} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}},`
                         :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]}, \text{kernel\_size[2]})`.
                         The values of these weights are sampled from
                         :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{2}\text{kernel\_size}[i]}`
        bias (Tensor):   the learnable bias of the module of shape (out_channels). If :attr:`bias` is ``True``,
                         then the values of these weights are
                         sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{2}\text{kernel\_size}[i]}`

    Examples::

        >>> # With square kernels and equal stride
        >>> m = nn.Conv3d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0))
        >>> input = torch.randn(20, 16, 10, 50, 100)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """


    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_count: int,
        stride: _size_3_t = 1,
        padding: _size_3_t = 0,
        dilated_kernel_size: _size_3_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
    ):
        stride_ = _triple(stride)
        padding_ = _triple(padding)
        dilated_kernel_size_ = _triple(dilated_kernel_size)
        super(Dcls3d, self).__init__(
            in_channels, out_channels, kernel_count, stride_, padding_, dilated_kernel_size_,
            False, _triple(0), groups, bias, padding_mode)

        self.DCK = ConstructKernel3d(self.out_channels, 
        			     self.in_channels, 
        			     self.groups, 
        			     self.kernel_count, 
        			     self.dilated_kernel_size)
    def extra_repr(self):
        s = super(Dcls3d, self).extra_repr()
        return s.format(**self.__dict__)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor], P: Tensor):
            if self.padding_mode != 'zeros':
                return F.conv3d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                self.DCK(weight, P), bias,
                                self.stride, _triple(0), _triple(1), self.groups)
            return F.conv3d(input, self.DCK(weight, P), bias,
                                   self.stride, self.padding, _triple(1), self.groups)

    def forward(self, input: Tensor) -> Tensor:
            return self._conv_forward(input, self.weight, self.bias, self.P)
