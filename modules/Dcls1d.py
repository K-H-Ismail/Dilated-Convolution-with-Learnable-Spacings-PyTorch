import math
import warnings

import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn.modules import Module

from functions.dcls_functionnal_full import SurrogateDilation1d


class Dcls1d(Module):

  def reset_parameters(self) -> None:      
      init.kaiming_uniform_(self.weight, a=math.sqrt(5))
      if self.bias is not None:
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)        

      P = torch.zeros(self.out_channels, self.in_channels // self.groups,self.kernel_size)
      for i in range(self.kernel_size):
        P[:,:,i] = -int(self.kernel_size * self.dilation /2) + i*self.dilation 

      self.P = torch.nn.Parameter(P, requires_grad=True)


  def __init__(self, 
               in_channels: int,
               out_channels: int,
               kernel_size,
               dilation, 
               bias=None,
               stride=1, 
               padding=0,  
               groups=1, 
               sigma=0.5, 
               overlapping="add", 
               border="clamp", 
               interpolation="subpixel") -> None:
    super(Dcls1d, self).__init__()
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
    self.sigma = sigma   
    self.overlapping = overlapping   
    self.border = border   
    self.interpolation = interpolation                           
    self.weight = Parameter(torch.Tensor(out_channels, in_channels // groups, self.kernel_size))
    self.P = Parameter(torch.Tensor(out_channels, in_channels // groups, self.kernel_size))

    
    self.reset_parameters()

  def forward(self, input: Tensor) -> Tensor:
    
        kernel = SurrogateDilation1d.apply(self.weight,
                                             self.P,
                                             self.dilation)


        return torch.nn.functional.conv1d(input,  
                                          kernel,          
                                          self.bias,                                           
                                          self.stride, 
                                          self.padding, 
                                          1,
                                          self.groups)

        

