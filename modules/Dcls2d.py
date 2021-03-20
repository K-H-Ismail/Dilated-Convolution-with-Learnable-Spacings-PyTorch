import math
import warnings

import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn.modules import Module

from functions.dcls_functionnal import SurrogateDilation

class Dcls2d(Module):

  def reset_parameters(self) -> None:      
      init.kaiming_uniform_(self.weight, a=math.sqrt(5))
      if self.bias is not None:
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)        

      P1 = torch.zeros(self.in_channels // self.groups,self.kernel_size[0],self.kernel_size[1])
      for i in range(self.kernel_size[0]):
        P1[:,i,:] = -int(self.kernel_size[0] * self.dilation[0] /2) + i*self.dilation[0]
        #if i in [0,1]:
        #    P1[:,i,:] = P1[:,i,:] + 0.5             
        #else:
        #    P1[:,i,:] = P1[:,i,:] - 0.5
      self.P1 = torch.nn.Parameter(P1, requires_grad=True)

      P2 = torch.zeros(self.in_channels // self.groups,self.kernel_size[0],self.kernel_size[1])
      for i in range(self.kernel_size[1]):
        P2[:,:,i] = -int(self.kernel_size[1] * self.dilation[1] /2) + i*self.dilation[1]

        #if i in [0,1] :
        #    P2[:,:,i] = P2[:,:,i] + 0.5             
        #else: 
         #   P2[:,:,i] = P2[:,:,i] - 0.5
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
    self.weight = Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
    self.P1 = Parameter(torch.Tensor(in_channels // groups, *self.kernel_size))
    self.P2 = Parameter(torch.Tensor(in_channels // groups, *self.kernel_size))
    
    self.reset_parameters()

  def forward(self, input: Tensor) -> Tensor:

    return  SurrogateDilation.apply(input,
                                    self.weight,
                                    self.P1,
                                    self.P2,
                                    self.bias,                                    
                                    self.dilation,                                    
                                    self.stride, 
                                    self.padding, 
                                    self.groups)
        

