import math
import warnings

import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn.modules import Module

from functions.dcls_functionnal import SurrogateDilation
#from .dcls_functionnal import SurrogateDilationLegacy

class Dcls2d(Module):

  def reset_parameters(self) -> None:      
      init.kaiming_uniform_(self.weight, a=math.sqrt(5))

      P1 = torch.zeros(self.kernel_size[0],self.kernel_size[1],self.out_channels,self.in_channels // self.groups)
      for i in range(self.kernel_size[0]):
        P1[i,:,:,:] = int((i*(self.dilation[0])-self.dilation[0]*self.kernel_size[0]//2 + self.dilation[0]//(2))) 
      self.P1 = torch.nn.Parameter(P1, requires_grad=True)

      P2 = torch.zeros(self.kernel_size[0],self.kernel_size[1],self.out_channels,self.in_channels // self.groups)
      for i in range(self.kernel_size[1]):
        P2[:,i,:,:] = int((i*(self.dilation[1])-self.dilation[1]*self.kernel_size[1]//2 + self.dilation[1]//(2))) 
      self.P2 = torch.nn.Parameter(P2, requires_grad=True)
      #init.uniform_(self.P1,a=-(self.kernel_size[0]*self.dilation[0])//2, b=(self.kernel_size[0]*self.dilation[0]-1)//2)
      #init.uniform_(self.P2,a=-(self.kernel_size[1]*self.dilation[1])//2, b=(self.kernel_size[1]*self.dilation[1]-1)//2)
  def __init__(self, 
               in_channels: int,
               out_channels: int,
               kernel_size,
               dilation, 
               bias=None, 
               stride=(1,1), 
               padding=(0,0),  
               groups=1, 
               sigma=0.5, 
               overlapping="add", 
               border="clamp", 
               interpolation="subpixel") -> None:
    super(Dcls2d, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size =  kernel_size
    self.dilation = dilation
    if bias:
        self.bias = Parameter(torch.Tensor(out_channels))
    else:
        self.register_parameter('bias', None)    
    self.stride = stride  
    self.padding = padding  
    self.groups = groups  
    self.sigma = sigma   
    self.overlapping = overlapping   
    self.border = border   
    self.interpolation = interpolation                           
    self.weight = Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
    self.P1 = Parameter(torch.Tensor(*self.kernel_size, out_channels, in_channels // groups))
    self.P2 = Parameter(torch.Tensor(*self.kernel_size, out_channels, in_channels // groups))
    
    self.reset_parameters()

  def forward(self, input: Tensor) -> Tensor:
    '''return torch.nn.functional.conv2d(input, 
                                    SurrogateDilationLegacy(self.weight,
                                                            self.P1,
                                                            self.P2,
                                                            ((self.kernel_size[0]-1)*self.dilation[0]+1,(self.kernel_size[1]-1)*self.dilation[1]+1,self.out_channels,self.in_channels//self.groups)), 
                                    self.bias, 
                                    self.stride, 
                                    self.padding, 
                                    (1,1), 
                                    self.groups)'''

    return torch.nn.functional.conv2d(input, 
                                    SurrogateDilation.apply(self.weight,
                                                            self.P1,
                                                            self.P2,
                                                            ((self.kernel_size[0]-1)*self.dilation[0]+1,(self.kernel_size[1]-1)*self.dilation[1]+1,self.out_channels,self.in_channels//self.groups)), 
                                    self.bias, 
                                    self.stride, 
                                    self.padding, 
                                    (1,1), 
                                    self.groups)
        
  def extra_repr(self):
      s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
            ', stride={stride}')
      if self.padding != (0,) * len(self.padding):
          s += ', padding={padding}'
      if self.dilation != (1,) * len(self.dilation):
          s += ', dilation={dilation}'
      if self.groups != 1:
          s += ', groups={groups}'
      if self.bias is None:
          s += ', bias=False'
      return s.format(**self.__dict__)        

