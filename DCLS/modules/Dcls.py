# coding=utf-8
import math
import warnings

import torch
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import DCLS
import DCLS.functions.dcls_functionnal as SD
from torch.nn import init
from torch.nn.modules import Module
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Optional, List, Tuple



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
                     'out_channels', 'kernel_count', 'scaling']
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
    scaling: float

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
                 padding_mode: str,
                 scaling: float) -> None:
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
        self.scaling = scaling
        self.padding_mode = padding_mode
        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, kernel_count))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, kernel_count))

        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.P = Parameter(torch.Tensor(len(dilated_kernel_size), in_channels // groups, kernel_count))
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
                init.normal_(self.P.select(0,i), 0, 0.5).clamp(-lim, lim).div_(self.scaling)

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
        if self.scaling != 1.0:
            s += ', scaling={scaling} (applied scaling)'
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



class Dcls2d(_DclsNd):
    __doc__ = r"""Applies a 2D dcls convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{\text{in}}, H, W)` and output :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})`
    can be precisely described as:

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, position(k))


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

    * :attr:`dilation` controls the maximum size of the dilated kernel.
      The spacings between the kernel points are learned in a gradient based manner.

    {groups_note}

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    Note:
        {depthwise_separable_note}


    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0
        padding_mode (string, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        dilation (int or tuple, optional): Maximum spacing between kernel elements. Default: 1
        sign_grad (bool, optional): Return the sign of the gradient for positions. Default: False
        gain (float, optional): Extra multiplicative factor for scaling. Default: 1.0
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
        P (Tensor): the learnable positions of the module of shape
            :math:`(\frac{\text{in\_channels}}{\text{groups}},`
            :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]})`.
            The values of these positions are sampled from
            :math:`\mathcal{N}(0, \sqrt{k})` where
            :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`

    Examples:

        >>> # With square kernels and equal stride
        >>> m = nn.Dcls2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.Dcls2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> # non-square kernels and unequal stride and with padding and learnable dilation
        >>> m = nn.Dcls2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
        >>> input = torch.randn(20, 16, 50, 100)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation
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
        scaling: float = 1.0
    ):
        stride_ = _pair(stride)
        padding_ = _pair(padding)
        dilated_kernel_size_ = _pair(dilated_kernel_size)
        super(Dcls2d, self).__init__(
            in_channels, out_channels, kernel_count, stride_, padding_, dilated_kernel_size_,
            False, _pair(0), groups, bias, padding_mode, scaling)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor], P1: Tensor, P2: Tensor):
        if self.padding_mode != 'zeros':
            return SD.dcls2d_conv.apply(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, P1, P2, bias, self.stride, _pair(0), self.dilated_kernel_size, self.groups, self.scaling)
        return SD.dcls2d_conv.apply(input, weight, P1, P2, bias,
                               self.stride, self.padding, self.dilated_kernel_size, self.groups, self.scaling)

    def forward(self, input: Tensor) -> Tensor:
            return self._conv_forward(input, self.weight, self.bias, self.P.select(0,0), self.P.select(0,1));



