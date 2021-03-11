import torch
import numpy as np

class SurrogateCeil(torch.autograd.Function):
    

    @staticmethod 
    def forward(ctx, input,n,m,sigma = 0.5):
        
        ctx.save_for_backward(input)
        ctx.n = n
        ctx.m = m
        ctx.sigma = sigma

        return input.ceil().clamp(-n,m)

    @staticmethod
    def backward(ctx, grad_output):
        
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()

        grad = torch.zeros_like(input)
        for k in range(-ctx.n+1,ctx.m+1,1):
            grad += torch.sigmoid(ctx.sigma*(input+k))*torch.sigmoid(-ctx.sigma*(input+k))

        return grad_input*grad, None, None, None



def SurrogateCeil_prime(input,n,m,sigma=0.5): 
    output = torch.zeros_like(input)
    for k in range(-n+1,m+1,1):
        output += torch.sigmoid(sigma*(input+k))*torch.sigmoid(-sigma*(input+k))

    return output


def make_matrix(dnext,dnext_t,mult,shapes):
  small_shapes = dnext.size()
  kernel_size1, kernel_size2, output_shape, input_shape = small_shapes[0], small_shapes[1], small_shapes[2], small_shapes[3]
  large_shapes = torch.Size(shapes)

  #mask = (dnext_t >= 0) *  (dnext_t <= shapes[1] - 1) * (dnext >= 0) * (dnext <= shapes[0] - 1)
  #dnext = dnext[mask] 
  #dnext_t = dnext_t[ mask] 

  indices = torch.zeros((kernel_size1* kernel_size2* output_shape* input_shape,4), device=dnext.device)
  indices[:,0] = dnext.reshape(-1).clamp(0,shapes[0]-1)
  indices[:,1] = dnext_t.reshape(-1).clamp(0,shapes[1]-1)
  indices[:,2:] = (torch.Tensor(np.indices((output_shape,input_shape)).reshape(2,output_shape*input_shape).T).repeat(kernel_size1*kernel_size2,1))#[mask.reshape(-1)]

  values = (mult).reshape(-1)

  sinput = torch.sparse.FloatTensor(indices.long().t(), values, large_shapes)
  return sinput

class SurrogateDilation_old(torch.autograd.Function):
    
    @staticmethod 
    def forward(ctx, weights, delays, delays_t, shapes):
        surr_ceil  = SurrogateCeil.apply

        delay_range1, delay_range2, output_shape, input_shape = shapes[0], shapes[1], shapes[2], shapes[3]
        
        output = torch.zeros((delay_range1, delay_range2, output_shape, input_shape),dtype=weights.dtype, device=weights.device)

        w_t = weights.permute(2,3,0,1)

        half_range_bot, half_range_top = (delay_range1)//2 ,  (delay_range1)//2-(delay_range1+1)%2 
        half_range_t_bot, half_range_t_top = (delay_range2)//2 ,  (delay_range2)//2-(delay_range2+1)%2 

        D_next_g = surr_ceil(delays,half_range_bot,half_range_top)
        rest_g = D_next_g-delays.clamp(min=-half_range_bot  , max=half_range_top)
        D_next_g_t = surr_ceil(delays_t,half_range_t_bot, half_range_t_top)
        rest_g_t = D_next_g_t-delays_t.clamp(min=-half_range_t_bot  , max=half_range_t_top)
                
        D_next_g += half_range_bot
        D_next_g_t += half_range_t_bot
        
        ctx.save_for_backward(weights, D_next_g, rest_g, D_next_g_t, rest_g_t)
        ctx.shapes = shapes
        
        '''for i in range(w_t.size(0)):
           for j in range(w_t.size(1)):

               D_next = (D_next_g[i,j,:,:]).long()
               rest = rest_g[i,j,:,:]
               D_next_t = (D_next_g_t[i,j,:,:]).long()
               rest_t = rest_g_t[i,j,:,:]


               output[D_next,D_next_t] = w_t[i,j,:,:] * (1 - rest)*(1 - rest_t)
               output[D_next-1,D_next_t] = w_t[i,j,:,:] * (rest)*(1 - rest_t)
               output[D_next,D_next_t-1] = w_t[i,j,:,:] * (1 - rest)*(rest_t)
               output[D_next-1,D_next_t-1] = w_t[i,j,:,:] * (rest)*(rest_t)

        return output.permute(2,3,0,1) '''
         
        
        sinput = make_matrix(D_next_g,D_next_g_t,w_t*(1 - rest_g)*(1 - rest_g_t),shapes) \
                + make_matrix(D_next_g-1,D_next_g_t,w_t*(rest_g)*(1 - rest_g_t),shapes) \
                + make_matrix(D_next_g,D_next_g_t-1,w_t*(1 - rest_g)*(rest_g_t),shapes) \
                + make_matrix(D_next_g-1,D_next_g_t-1,w_t*(rest_g)*(rest_g_t),shapes) 
        dinput = sinput.to_dense()                   
        return dinput.permute(2,3,0,1) 


    @staticmethod
    def backward(ctx, grad_output):
        
        delay_range1, delay_range2, output_shape, input_shape = ctx.shapes[0], ctx.shapes[1], ctx.shapes[2], ctx.shapes[3]             
        weights, D_next_g, rest_g, D_next_g_t, rest_g_t = ctx.saved_tensors

        half_range_bot, half_range_top = (delay_range1)//2 ,  (delay_range1)//2-(delay_range1+1)%2 
        half_range_t_bot, half_range_t_top = (delay_range2)//2 ,  (delay_range2)//2-(delay_range2+1)%2 
        
        w_t = weights.permute(2,3,0,1)
        
        grad_output_t = grad_output.permute(2,3,0,1)
        grad_weights = torch.zeros_like(w_t)
        grad_delays = torch.zeros_like(w_t)
        grad_delays_t = torch.zeros_like(w_t)
        
        for i in range(w_t.size(0)):
           for j in range(w_t.size(1)):

               D_next = (D_next_g[i,j,:,:]).long()
               rest = rest_g[i,j,:,:]
               D_next_t = (D_next_g_t[i,j,:,:]).long()
               rest_t = rest_g_t[i,j,:,:]

               D_next1 = (D_next-1).clamp(0,delay_range1-1)   
               D_next_t1 = (D_next_t-1).clamp(0,delay_range2-1)

               Back_D_next = SurrogateCeil_prime(D_next_g[i,j,:,:],half_range_bot, half_range_top)
               Back_D_next_t = SurrogateCeil_prime(D_next_g_t[i,j,:,:],half_range_t_bot, half_range_t_top)
            
               grad_weights_tmp = grad_output_t[D_next,D_next_t] * (1 - rest)*(1 - rest_t)\
                                       + grad_output_t[D_next1,D_next_t] * (rest)*(1 - rest_t)\
                                       + grad_output_t[D_next,D_next_t1] * (1 - rest)*(rest_t)\
                                       + grad_output_t[D_next1,D_next_t1] * (rest)*(rest_t)
                
               grad_delays_tmp = grad_output_t[D_next,D_next_t] * w_t[i,j,:,:] * (1 - Back_D_next)*(1 - rest_t)\
                                       + grad_output_t[D_next1,D_next_t] * w_t[i,j,:,:] * (Back_D_next - 1)*(1 - rest_t)\
                                       + grad_output_t[D_next,D_next_t1] * w_t[i,j,:,:] * (1 - Back_D_next)*(rest_t)\
                                       + grad_output_t[D_next1,D_next_t1] * w_t[i,j,:,:] * (Back_D_next - 1)*(rest_t) 
            
               grad_delays_t_tmp = grad_output_t[D_next,D_next_t] * w_t[i,j,:,:] * (1 - rest)*(1 - Back_D_next_t)\
                                       + grad_output_t[D_next1,D_next_t] * w_t[i,j,:,:] * (rest)*(1 - Back_D_next_t)\
                                       + grad_output_t[D_next,D_next_t1] * w_t[i,j,:,:] * (1 - rest)*(Back_D_next_t - 1)\
                                       + grad_output_t[D_next1,D_next_t1] * w_t[i,j,:,:] * (rest)*(Back_D_next_t - 1)         
                
               grad_weights[i,j,:,:] = grad_weights_tmp[0,0,:,:]   
               grad_delays[i,j,:,:] = grad_delays_tmp[0,0,:,:]   
               grad_delays_t[i,j,:,:] = grad_delays_t_tmp[0,0,:,:]   
      
        return grad_weights.permute(2,3,0,1), grad_delays, grad_delays_t, None

class SurrogateHeaviside(torch.autograd.Function):
    
    # Activation function with surrogate gradient
    sigma = 0.5

    @staticmethod 
    def forward(ctx, input):
        
        output = torch.zeros_like(input)
        output[input >= 0] = 1.0
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # approximation of the gradient using sigmoid function
        grad = grad_input*torch.sigmoid(SurrogateHeaviside.sigma*input)*torch.sigmoid(-SurrogateHeaviside.sigma*input)
    
def SurrogateLeq(input1,input2):
    surr_heaviside = SurrogateHeaviside.apply
    return surr_heaviside(input1-input2)

def SurrogateEq(input1,input2):

    return SurrogateLeq(input1,input2) * SurrogateLeq(input2,input1)    
    
    
def SurrogateDilationLegacy(weights, delays, delays_t, shapes):
    surr_ceil  = SurrogateCeil.apply
    delay_range1, delay_range2, output_shape, input_shape = shapes[0], shapes[1], shapes[2], shapes[3]

    output = torch.zeros((delay_range1, delay_range2, output_shape, input_shape),dtype=weights.dtype, device=weights.device)

    w_t = weights.permute(2,3,0,1)

    half_range_bot, half_range_top = (delay_range1)//2 ,  (delay_range1)//2-(delay_range1+1)%2 
    half_range_t_bot, half_range_t_top = (delay_range2)//2 ,  (delay_range2)//2-(delay_range2+1)%2 

    D_next_g = surr_ceil(delays,half_range_bot,half_range_top)
    rest_g = D_next_g-delays.clamp(min=-half_range_bot  , max=half_range_top)
    D_next_g_t = surr_ceil(delays_t,half_range_t_bot, half_range_t_top)
    rest_g_t = D_next_g_t-delays_t.clamp(min=-half_range_t_bot  , max=half_range_t_top)

    for i in range(delay_range1):
      for j in range(delay_range2):

         i_centered = int(i - half_range_top)
         j_centered = int(j - half_range_t_top)   
         i_w = int(i*w_t.size(0)/delay_range1)
         j_w = int(j*w_t.size(1)/delay_range2)

         D_next = D_next_g[i_w,j_w,:,:]
         rest = rest_g[i_w,j_w,:,:]
         D_next_t = D_next_g_t[i_w,j_w,:,:]
         rest_t = rest_g_t[i_w,j_w,:,:]

         mask_next = SurrogateEq(D_next,i_centered)*SurrogateEq(D_next_t,j_centered)
         mask_prev = SurrogateEq(D_next,i_centered+1)*SurrogateEq(D_next_t,j_centered)
         mask_prev_t = SurrogateEq(D_next,i_centered)*SurrogateEq(D_next_t,j_centered+1)                    
         mask_next_t = SurrogateEq(D_next,i_centered+1)*SurrogateEq(D_next_t,j_centered+1)   

         masked_next_kernel = output[i, j,:,:] + (w_t[i_w,j_w,:,:] * (1 - rest)*(1 - rest_t)  - output[i, j,:,:])*mask_next 
         output[i, j,:,:] = masked_next_kernel                        
         masked_prev_kernel = output[i, j,:,:] + (w_t[i_w,j_w,:,:] * rest*(1 - rest_t)  - output[i, j,:,:])*mask_prev 
         output[i, j,:,:] = masked_prev_kernel
         masked_prev_kernel_t = output[i, j,:,:] + (w_t[i_w,j_w,:,:] * rest_t*(1 - rest) - output[i, j,:,:])*mask_prev_t 
         output[i, j,:,:] = masked_prev_kernel_t    
         masked_next_kernel_t = output[i, j,:,:] + (w_t[i_w,j_w,:,:] * rest*rest_t     - output[i, j,:,:])*mask_next_t 
         output[i, j,:,:] = masked_next_kernel_t                        
    return output.permute(2,3,0,1)     
