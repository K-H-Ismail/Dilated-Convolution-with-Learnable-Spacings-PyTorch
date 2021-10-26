# coding=utf-8
import math
import warnings

import torch
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import DCLS
import DCLS.construct.functions.dcls_functionnal as SD
import DCLS.construct.functions.swc_functionnal as SW
from torch.nn import init
from torch.nn.modules import Module
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Optional, List, Tuple
import operator


# overloading + and // operators for Union(int,Tuple[int,...])
class _size_1_op_t:
    def __init__(self, a: _size_1_t):
        self.matter = a
        self.type = type(a)
 
    def _op_tuple(self, a: tuple, b: tuple, binary_op):
        return _size_1_op_t((binary_op(a[0],b[0]),))
    
    def _op_tuple_int(self, a: tuple, b: int, binary_op):
        return _size_1_op_t((binary_op(a[0],b),))
    
    def _op_int_tuple(self, a: int, b: tuple, binary_op):
        return _size_1_op_t((binary_op(a,b[0]),))
    
    def _op_int(self, a: int, b: int, binary_op):
        return _size_1_op_t(binary_op(a,b))    
    
    def __add__(self, other: _size_1_t):      
        if (self.type == tuple and other.type == tuple):
            return self._op_tuple(self.matter, other.matter, operator.__add__)            
        elif (self.type == tuple and other.type == int):
            return self._op_tuple_int(self.matter, other.matter, operator.__add__)            
        elif (self.type == int and other.type == tuple):
            return self._op_int_tuple(self.matter, other.matter, operator.__add__)          
        else:
            return self._op_int(self.matter, other.matter, operator.__add__)
        
    def __floordiv__(self, other: _size_1_t):
        if (self.type == tuple and other.type == tuple):
            return self._op_tuple(self.matter, other.matter, operator.__floordiv__)            
        elif (self.type == tuple and other.type == int):
            return self._op_tuple_int(self.matter, other.matter, operator.__floordiv__)            
        elif (self.type == int and other.type == tuple):
            return self._op_int_tuple(self.matter, other.matter, operator.__floordiv__)          
        else:
            return self._op_int(self.matter, other.matter, operator.__floordiv__)
        
    def get(self):
        return _single(self.matter)
        
class _size_2_op_t:
    def __init__(self, a: _size_2_t):
        self.matter = a
        self.type = type(a)
 
    def _op_tuple(self, a: tuple, b: tuple, binary_op):
        return _size_2_op_t((binary_op(a[0],b[0]), binary_op(a[1],b[1])))
    
    def _op_tuple_int(self, a: tuple, b: int, binary_op):
        return _size_2_op_t((binary_op(a[0],b), binary_op(a[1],b)))
    
    def _op_int_tuple(self, a: int, b: tuple, binary_op):
        return _size_2_op_t((binary_op(a,b[0]), binary_op(a,b[1])))
    
    def _op_int(self, a: int, b: int, binary_op):
        return _size_2_op_t(binary_op(a,b))     
    
    def __add__(self, other: _size_2_t):
        if (self.type == tuple and other.type == tuple):
            return self._op_tuple(self.matter, other.matter, operator.__add__)            
        elif (self.type == tuple and other.type == int):
            return self._op_tuple_int(self.matter, other.matter, operator.__add__)            
        elif (self.type == int and other.type == tuple):
            return self._op_int_tuple(self.matter, other.matter, operator.__add__)          
        else:
            return self._op_int(self.matter, other.matter, operator.__add__)  
        
    def __floordiv__(self, other: _size_2_t):
        if (self.type == tuple and other.type == tuple):
            return self._op_tuple(self.matter, other.matter, operator.__floordiv__)            
        elif (self.type == tuple and other.type == int):
            return self._op_tuple_int(self.matter, other.matter, operator.__floordiv__)            
        elif (self.type == int and other.type == tuple):
            return self._op_int_tuple(self.matter, other.matter, operator.__floordiv__)          
        else:
            return self._op_int(self.matter, other.matter, operator.__floordiv__)
        
    def get(self):
        return _pair(self.matter)        
        
class _size_3_op_t:
    def __init__(self, a: _size_3_t):
        self.matter = a
        self.type = type(a)
 
    def _op_tuple(self, a: tuple, b: tuple, binary_op):
        return _size_2_op_t((binary_op(a[0],b[0]), binary_op(a[1],b[1]), binary_op(a[2],b[2])))
    
    def _op_tuple_int(self, a: tuple, b: int, binary_op):
        return _size_2_op_t((binary_op(a[0],b), binary_op(a[1],b), binary_op(a[2],b)))
    
    def _op_int_tuple(self, a: int, b: tuple, binary_op):
        return _size_2_op_t((binary_op(a,b[0]), binary_op(a,b[1]), binary_op(a,b[2])))
    
    def _op_int(self, a: int, b: int, binary_op):
        return _size_3_op_t(binary_op(a,b))     
    
    def __add__(self, other: _size_3_t):
        if (self.type == tuple and other.type == tuple):
            return self._op_tuple(self.matter, other.matter, operator.__add__)            
        elif (self.type == tuple and other.type == int):
            return self._op_tuple_int(self.matter, other.matter, operator.__add__)            
        elif (self.type == int and other.type == tuple):
            return self._op_int_tuple(self.matter, other.matter, operator.__add__)          
        else:
            return self._op_int(self.matter, other.matter, operator.__add__) 
        
    def __floordiv__(self, other: _size_3_t):
        if (self.type == tuple and other.type == tuple):
            return self._op_tuple(self.matter, other.matter, operator.__floordiv__)            
        elif (self.type == tuple and other.type == int):
            return self._op_tuple_int(self.matter, other.matter, operator.__floordiv__)            
        elif (self.type == int and other.type == tuple):
            return self._op_int_tuple(self.matter, other.matter, operator.__floordiv__)          
        else:
            return self._op_int(self.matter, other.matter, operator.__floordiv__)
        
    def get(self):
        return _triple(self.matter)        
        
    
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

    __constants__ = ['stride', 'padding', 'dilation', 'groups',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size', 'gain']
    __annotations__ = {'bias': Optional[torch.Tensor]}

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
        ...

    _in_channels: int
    out_channels: int
    kernel_size: Tuple[int, ...]
    stride: Tuple[int, ...]
    padding: Tuple[int, ...]
    dilation: Tuple[int, ...]
    transposed: bool
    output_padding: Tuple[int, ...]
    groups: int
    padding_mode: str
    weight: Tensor
    bias: Optional[Tensor]      
    gain: float 
        
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, ...],
                 stride: Tuple[int, ...],
                 padding: Tuple[int, ...],
                 dilation: Tuple[int, ...],
                 transposed: bool,
                 output_padding: Tuple[int, ...],
                 groups: int,
                 bias: bool,
                 padding_mode: str,
                 gain: float) -> None:
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
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.gain = gain        
        self.padding_mode = padding_mode
        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.bias = Parameter(torch.Tensor(out_channels))            
            #self.register_parameter('bias', None)
        self.P = Parameter(torch.Tensor(len(kernel_size), out_channels, in_channels // groups, *kernel_size))          
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        #init.zeros_(self.P)     
        for i in range(len(self.kernel_size)):
            std = (self.dilation[i]/2) / (self.gain * math.sqrt(self.out_channels * (self.in_channels // self.groups) * self.kernel_size[0] * self.kernel_size[1]))             
            init.normal_(self.P.select(0,i), 0.0, std)     
        
        
    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation): 
            s += ', dilation_max={dilation} (learnable)'
        if self.gain != 1.0: 
            s += ', gain={gain} (an extra multiplicative factor is applied to scaling)'             
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

class _DclsN_Md(Module):

    __constants__ = ['dim_dilation', 'stride', 'padding', 'dilation', 'groups',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']
    __annotations__ = {'bias': Optional[torch.Tensor]}

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
        ...

    _in_channels: int
    out_channels: int
    kernel_size: Tuple[int, ...]
    stride: Tuple[int, ...]
    padding: Tuple[int, ...]
    dilation: Tuple[int, ...]
    transposed: bool
    output_padding: Tuple[int, ...]
    groups: int
    padding_mode: str
    weight: Tensor
    bias: Optional[Tensor]
    dim_dilation: Tuple[int, ...]       

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, ...],
                 stride: Tuple[int, ...],
                 padding: Tuple[int, ...],
                 dilation: Tuple[int, ...],
                 dim_dilation: Tuple[int, ...],                 
                 transposed: bool,
                 output_padding: Tuple[int, ...],
                 groups: int,
                 bias: bool,
                 padding_mode: str) -> None:
        super(_DclsN_Md, self).__init__()
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
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.dim_dilation = dim_dilation        
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
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.P = Parameter(torch.Tensor(len(dim_dilation), out_channels, in_channels // groups, *kernel_size))          
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        #init.zeros_(self.P)
        for i in range(len(self.dim_dilation)): 
            init.kaiming_uniform_(self.P[i,:], a=math.sqrt(5))        

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation): 
            s += ', dilation_max={dilation} (learnable along {dim_dilation})'
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
        super(_DclsN_Md, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'


class Dcls1d(_DclsNd):
    __doc__ = r"""Applies a 1D convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{\text{in}}, L)` and output :math:`(N, C_{\text{out}}, L_{\text{out}})` can be
    precisely described as:

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{in} - 1} \text{weight}(C_{\text{out}_j}, k)
        \star \text{input}(N_i, k)

    where :math:`\star` is the valid `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`L` is a length of signal sequence.
    """ + r"""

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    * :attr:`stride` controls the stride for the cross-correlation, a single
      number or a one-element tuple.

    * :attr:`padding` controls the amount of implicit padding on both sides
      for :attr:`padding` number of points.

    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.

    {groups_note}

    Note:
        {depthwise_separable_note}
    Note:
        {cudnn_reproducibility_note}

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0
        padding_mode (string, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        dilation (int or tuple, optional): Spacing between kernel
            elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``

    """ + r"""

    Shape:
        - Input: :math:`(N, C_{in}, L_{in})`
        - Output: :math:`(N, C_{out}, L_{out})` where

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
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: _size_1_t = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
    ):           
        # we create new variables below to make mypy happy since kernel_size has
        # type Union[int, Tuple[int]] and kernel_size_ has type Tuple[int]
        kernel_size_ = _single(kernel_size)
        stride_ = _single(stride)
        padding_ = (_size_1_op_t(padding) + _size_1_op_t(dilation) // _size_1_op_t(2)).get()
        dilation_ = _single(dilation)
        super(Dcls1d, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _single(0), groups, bias, padding_mode)
    
    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor], P: Tensor):
        if self.padding_mode != 'zeros':
            return F.conv1d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            SD.ConstructKernel1d.apply(weight, P, self.dilation), bias, self.stride,
                            _single(0), _single(1), self.groups)
        return F.conv1d(input, SD.ConstructKernel1d.apply(weight, P, self.dilation), bias, self.stride,
                        self.padding, _single(1), self.groups)

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias, self.P.select(0,0))


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

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

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
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0
        padding_mode (string, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
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
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        is_swc: bool = False,
        chunk_size: int = 1024,
        gain: float = 1.0        
    ):            
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = _pair(padding)
        dilation_ = _pair(dilation)
        super(Dcls2d, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode, gain)
        self.is_swc = is_swc
        self.chunk_size = chunk_size        
        
    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor], P1: Tensor, P2: Tensor):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            SD.ConstructKernel2d.apply(weight, P1, P2, self.dilation, self.gain), bias, self.stride,
                            _pair(0), _pair(1), self.groups)
        return F.conv2d(input, SD.ConstructKernel2d.apply(weight, P1, P2, self.dilation, self.gain), bias, self.stride,
                        self.padding, _pair(1), self.groups)
    
    def _conv_forward_swc(self, input: Tensor, weight: Tensor, bias: Optional[Tensor], P1: Tensor, P2: Tensor):
        if self.padding_mode != 'zeros':
            return SW.swc2d.apply(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            SD.ConstructKernel2d.apply(weight, P1, P2, self.dilation), bias, self.stride,
                            _pair(0), _pair(1), self.groups)
        return SW.swc2d.apply(input, SD.ConstructKernel2d.apply(weight, P1, P2, self.dilation), bias, self.stride,
                        self.padding, _pair(1), self.groups)

    def forward(self, input: Tensor) -> Tensor:
        if (self.dilation[0] * self.dilation[1] == 1) :
            return F.conv2d(input, self.weight, self.bias, self.stride, self.padding, _pair(1), self.groups)
            #return self._conv_forward_swc(input, self.weight, self.bias, self.P.select(0,0), self.P.select(0,1))
        else:
            #return self._conv_forward(input, self.weight, self.bias, self.P.select(0,0), self.P.select(0,1))
                
            return self._conv_forward(input, self.weight, self.bias, self.P.select(0,0), self.P.select(0,1));
    


    
class Dcls3d(_DclsNd):
    __doc__ = r"""Applies a 3D convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C_{in}, D, H, W)`
    and output :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` can be precisely described as:

    .. math::
        out(N_i, C_{out_j}) = bias(C_{out_j}) +
                                \sum_{k = 0}^{C_{in} - 1} weight(C_{out_j}, k) \star input(N_i, k)

    where :math:`\star` is the valid 3D `cross-correlation`_ operator
    """ + r"""

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    * :attr:`stride` controls the stride for the cross-correlation.

    * :attr:`padding` controls the amount of implicit padding on both
      sides for :attr:`padding` number of points for each dimension.

    * :attr:`dilation` controls the spacing between the kernel points; also known as the à trous algorithm.
      It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

    {groups_note}

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the depth, height and width dimension
        - a ``tuple`` of three ints -- in which case, the first `int` is used for the depth dimension,
          the second `int` for the height dimension and the third `int` for the width dimension

    Note:
        {depthwise_separable_note}

    Note:
        {cudnn_reproducibility_note}

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to all three sides of the input. Default: 0
        padding_mode (string, optional): ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    """ + r"""

    Shape:
        - Input: :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` where

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
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: _size_3_t = 0,
        dilation: _size_3_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'
    ):        
        kernel_size_ = _triple(kernel_size)
        stride_ = _triple(stride)
        padding_ = (_size_3_op_t(padding) + _size_3_op_t(dilation) // _size_3_op_t(2)).get()
        dilation_ = _triple(dilation)
        super(Dcls3d, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_, 
            False, _triple(0), groups, bias, padding_mode)
        
    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor], P1: Tensor, P2: Tensor, P3: Tensor):
        if self.padding_mode != 'zeros':
            return F.conv3d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            SD.ConstructKernel3d.apply(weight, P1, P2, P3, self.dilation), bias, self.stride,
                            _triple(0), _triple(1), self.groups)
        return F.conv3d(input, SD.ConstructKernel3d.apply(weight, P1, P2, P3, self.dilation), bias, self.stride,
                        self.padding, _triple(1), self.groups)

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias, self.P.select(0,0), self.P.select(0,1), self.P.select(0,2)) 



    
class Dcls2_1d(_DclsN_Md):
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

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

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
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0
        padding_mode (string, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
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
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        dim_dilation: _size_1_t = 0,        
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'  # TODO: refine this type
    ):
        
        def _adjust_padding(padding, dilation):
            if (type(padding) == tuple):
                return (padding[0] + dilation[0] // 2, padding[1])
            else:
                return _pair(padding + dilation // 2)
        
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = _adjust_padding(padding, dilation)
        dilation_ = _pair(dilation)
        super(Dcls2_1d, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_, dim_dilation,
            False, _pair(0), groups, bias, padding_mode)
        
    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor], P1: Tensor):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            SD.ConstructKernel2_1d.apply(weight, P1, self.dilation), bias, self.stride,
                            _pair(0), _pair(1), self.groups)
        return F.conv2d(input, SD.ConstructKernel2_1d.apply(weight, P1, self.dilation), bias, self.stride,
                        self.padding, _pair(1), self.groups)

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias, self.P.select(0,0)) 
                      
                      
                      
class Dcls3_1d(_DclsN_Md):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: _size_3_t = 0,
        dilation: _size_3_t = 1,
        dim_dilation: _size_1_t = 0,         
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'
    ):

        def _adjust_padding(padding, dilation):
            if (type(padding) == tuple):
                return (padding[0] + dilation[0] // 2, padding[1], padding[2])
            else:
                return _triple(padding + dilation // 2)
        
        kernel_size_ = _triple(kernel_size)
        stride_ = _triple(stride)
        padding_ = _adjust_padding(padding, dilation)
        dilation_ = _triple(dilation)
        super(Dcls3_1d, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_, dim_dilation,
            False, _triple(0), groups, bias, padding_mode)

    
    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor], P1: Tensor):
        if self.padding_mode != 'zeros':
            return F.conv3d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            SD.ConstructKernel3_1d.apply(weight, P1, self.dilation), bias, self.stride,
                            _triple(0), _triple(1), self.groups)
        return F.conv3d(input, SD.ConstructKernel3_1d.apply(weight, P1, self.dilation), bias, self.stride,
                        self.padding, _triple(1), self.groups)

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias, self.P.select(0,0))
    
    
class Dcls3_2d(_DclsN_Md):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: _size_3_t = 0,
        dilation: _size_3_t = 1,
        dim_dilation: _size_2_t = (0,1),         
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'
    ):

        def _adjust_padding(padding, dilation):
            if (type(padding) == tuple):
                return (padding[0] + dilation[0] // 2, padding[1] + dilation[1] // 2, padding[2])
            else:
                return _triple(padding + dilation // 2)
        
        kernel_size_ = _triple(kernel_size)
        stride_ = _triple(stride)
        padding_ = _adjust_padding()
        dilation_ = _triple(dilation)
        super(Dcls3_1d, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_, dim_dilation,
            False, _triple(0), groups, bias, padding_mode)
    
    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor], P1: Tensor, P2: Tensor):
        if self.padding_mode != 'zeros':
            return F.conv3d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            SD.ConstructKernel3_2d.apply(weight, P1, P2, self.dilation), bias, self.stride,
                            _triple(0), _triple(1), self.groups)
        return F.conv3d(input, SD.ConstructKernel3_2d.apply(weight, P1, P2, self.dilation), bias, self.stride,
                        self.padding, _triple(1), self.groups)

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias, self.P.select(0,0), self.P.select(0,1))    
    
    

                
