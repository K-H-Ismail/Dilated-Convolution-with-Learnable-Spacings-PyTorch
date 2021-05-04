import math
import warnings

import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn.modules import Module

from functions.swc_functionnal import swc2d

class Swc2d(Module):

  def reset_parameters(self) -> None:      
      init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
      if self.bias is not None:
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)        

  def __init__(self, 
               in_channels: int,
               out_channels: int,
               kernel_size,
               dilation, 
               bias=None, 
               stride=(1,1), 
               padding=(0,0),  
               groups=1,
               sparse_mm=False) -> None:
    super(Swc2d, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size =  kernel_size
    self.dilation = dilation
    self.bias = Parameter(torch.zeros(out_channels))
    
    if bias is None:
        self.bias.data = torch.zeros(out_channels).data
        self.bias.requires_grad = False
        
    self.stride = stride  
    self.padding = padding  
    self.groups = groups                             
    self.weight = Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
    self.sparse_mm = sparse_mm

    self.reset_parameters()

  def forward(self, input: Tensor) -> Tensor:

       
      return swc2d.apply(input,
                         self.weight,
                         self.bias,                     
                         self.dilation,                                    
                         self.stride, 
                         self.padding, 
                         self.groups,
                         self.sparse_mm)
