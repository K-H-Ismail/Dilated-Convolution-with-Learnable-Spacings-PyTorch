import math
import warnings

import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn.modules import Module

from functions.dcls_functionnal_full import SurrogateDilationFull
from functions.swc_functionnal import swc2d

class Dcls2dFull(Module):

  def reset_parameters(self) -> None:      
      init.kaiming_uniform_(self.weight, a=math.sqrt(5))
      if self.bias is not None:
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)        

      P1 = torch.zeros(self.out_channels, self.in_channels // self.groups,self.kernel_size[0],self.kernel_size[1])
      for i in range(self.kernel_size[0]):
        P1[:,:,i,:] = -int(self.kernel_size[0] * self.dilation[0] /2) + i*self.dilation[0] 
        
        #if i in [0,1]:
        #    P1[:,:,i,:] = P1[:,:,i,:] + 0.2             
        #else:
        #    P1[:,:,i,:] = P1[:,:,i,:] - 0.8
      self.P1 = torch.nn.Parameter(P1, requires_grad=True)

      P2 = torch.zeros(self.out_channels, self.in_channels // self.groups,self.kernel_size[0],self.kernel_size[1])
      for i in range(self.kernel_size[1]):
        P2[:,:,:,i] = -int(self.kernel_size[1] * self.dilation[1] /2) + i*self.dilation[1]  

        #if i in [0,1] :
        #    P2[:,:,:,i] = P2[:,:,:,i] + 0.25             
        #else: 
        #    P2[:,:,:,i] = P2[:,:,:,i] - 0.75
      self.P2 = torch.nn.Parameter(P2, requires_grad=True)

  def __init__(self, 
               in_channels: int,
               out_channels: int,
               kernel_size,
               dilation, 
               bias=None,
               sparse_mm=False,
               stride=(1,1), 
               padding=(0,0),  
               groups=1, 
               sigma=0.5, 
               overlapping="add", 
               border="clamp", 
               interpolation="subpixel") -> None:
    super(Dcls2dFull, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size =  kernel_size
    self.dilation = dilation
    self.bias = Parameter(torch.zeros(out_channels))
    self.sparse_mm = sparse_mm
    
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
    self.P1 = Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
    self.P2 = Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
    
    self.reset_parameters()

  def forward(self, input: Tensor) -> Tensor:
    
        kernel = SurrogateDilationFull.apply(self.weight,
                                             self.P1,
                                             self.P2,
                                             self.dilation)

        #print("Density:", kernel.nonzero().size(0)*100/kernel.numel(),"%")


        '''return swc2d.apply(input,  
                                   kernel,          
                                   self.bias, 
                                   (1,1),                                   
                                   self.stride, 
                                   self.padding, 
                                   self.groups,
                                   self.sparse_mm)'''

        return torch.nn.functional.conv2d(input,  
                              kernel,          
                              self.bias,                                           
                              self.stride, 
                              self.padding, 
                              (1,1),
                              self.groups)

        

