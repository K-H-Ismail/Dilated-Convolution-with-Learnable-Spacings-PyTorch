#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <math.h>


#define CUDA_KERNEL_LOOP(i, n)                                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);                 \
       i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
const int kMaxGridNum = 65535;
inline int GET_BLOCKS(const int N) {
  return std::min(kMaxGridNum, (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);
}

struct d_sigmoid {
  torch::Tensor operator() (torch::Tensor z, torch::Tensor sigma) 
  {
      auto s = at::sigmoid(sigma * z);
      return (1.0 - s) * s * sigma;   
  }
};
struct d_tanh {
    torch::Tensor operator() (torch::Tensor z, torch::Tensor sigma) 
    {
        auto s = at::tanh(z * sigma);   
        return 1.0 - s * s;   
    }
};
struct d_erf {
    torch::Tensor operator() (torch::Tensor z, torch::Tensor sigma) 
    {
        return M_2_SQRTPI * at::exp(- sigma * sigma * z * z); 
    }
};  
struct d_atan {
    torch::Tensor operator() (torch::Tensor z, torch::Tensor sigma) 
    {
        return 1.0 / (1.0 + sigma * sigma * z * z);   
    }
};

struct d_zero {
    torch::Tensor operator() (torch::Tensor z, torch::Tensor sigma) 
    {
        return torch::zeros_like(z);   
    }    
}; 

struct d_one {
    torch::Tensor operator() (torch::Tensor z, torch::Tensor sigma) 
    {
        return torch::ones_like(z);   
    }    
}; 

template <typename F> 
torch::Tensor d_ceil(torch::Tensor z, torch::Tensor sigma, const int bot, const int top, F d_activation) {
  auto s = torch::zeros_like(z);
  for (int i = 1-bot; i < top+1; i++) 
  { 
      s += d_activation(z + static_cast<double>(i), sigma);
  }
  return s;
}

template <typename F> 
torch::Tensor d_floor(torch::Tensor z, torch::Tensor sigma, const int bot, const int top, F d_activation) {
  auto s = torch::zeros_like(z);
  for (int i = -bot; i < top; i++) 
  { 
      s += d_activation(z + static_cast<double>(i), sigma);
  }
  return s;
}