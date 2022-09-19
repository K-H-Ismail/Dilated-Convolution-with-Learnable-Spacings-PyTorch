# Dilated-Convolution-with-Learnable-Spacings-PyTorch
Ismail Khalfaoui Hassani, Thomas Pellegrini and Timothée Masquelier

Dilated Convolution with Learnable Spacings (abbreviated to DCLS) is a novel convolution method based on gradient descent and interpolation. It could be seen as an improvement of the well known dilated convolution that has been widely explored in deep convolutional neural networks and which aims to inflate the convolutional kernel by inserting spaces between the kernel elements. 

In DCLS, the positions of the weights within the convolutional kernel are learned in a gradient-based manner, and the inherent problem of non-differentiability due to the integer nature of the positions in the kernel is solved by taking advantage of an interpolation method. 

For now, the code has only been implemented on [PyTorch](https://pytorch.org/), using Pytorch's C++ API and custom cuda extensions. 

- [Installation](#installation)
- [Usage](#usage)
- [Device Supports](#device-supports)
- [Publications and Citation](#publications-and-citation)
- [Contribution](#contribution)

The method is described in the arXiv preprint [Dilated Convolution with Learnable Spacings](https://arxiv.org/abs/2112.03740).
## Installation

DCLS is based on PyTorch and CUDA. Please make sure that you have installed all the requirements before you install DCLS.


**Install the last stable version from** [**PyPI**](https://pypi.org/project/DCLS/):

```bash
coming soon
```

**Install the latest developing version from the source codes**:

From [GitHub](https://github.com/K-H-Ismail/Dilated-Convolution-with-Learnable-Spacings-PyTorch):
```bash
git clone https://github.com/K-H-Ismail/Dilated-Convolution-with-Learnable-Spacings-PyTorch.git
cd Dilated-Convolution-with-Learnable-Spacings-PyTorch
python ./setup.py install --user --no-cache-dir
```


## Usage
Dcls methods could be easily used as a substitue of Pytorch's nn.Conv**n**d classical convolution method:

```python
from DCLS.construct.modules.Dcls import  Dcls2d

# With square kernels, equal stride and dilation
m = Dcls2d(16, 33, kernel_count=3, dilated_kernel_size=7).cuda()
input = torch.randn(20, 16, 50, 100).cuda()
output = m(input)
loss = output.sum()
loss.backward()
print(output, m.weight.grad, m.P.grad)
```

## Device Supports
DCLS only supports Nvidia CUDA GPU devices for the moment. The CPU version has not been implemented yet.

-   [x] Nvidia GPU
-   [ ] CPU

Make sure to have your data and model on CUDA GPU.
DCLS-im2col doesn't support mixed precision operations for now. By default every tensor is converted to have float32 precision.

## Publications and Citation

If you use DCLS in your work, please consider to cite it as follows:

```
@article{khalfaoui2021dilated,
  title={Dilated convolution with learnable spacings},
  author={Khalfaoui-Hassani, Ismail and Pellegrini, Thomas and Masquelier, Timoth{\'e}e},
  journal={arXiv preprint arXiv:2112.03740},
  year={2021}
}

```

## Contribution

This project is open source, therefore all your contributions are welcomed, whether it's reporting issues, finding and fixing bugs, requesting new features, and sending pull requests ...


