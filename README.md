[![arXiv](https://img.shields.io/badge/arXiv-2112.03740v4-b31b1b.svg?style=plastic)](https://arxiv.org/abs/2112.03740v4)
[![arXiv](https://img.shields.io/badge/arXiv-2306.00817v2-b31b1b.svg?style=plastic)](https://arxiv.org/abs/2306.00817v2)
[![medium](https://img.shields.io/badge/Medium-12100E?style=plastic)](https://medium.com/@khalfaoui.ismail/dea93c490a82)
[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nr2zF-Zw8QRuKN68Mfdw5uCtznBa-UpM?usp=sharing) 
# Dilated-Convolution-with-Learnable-Spacings-PyTorch

<div align=center>
<img src='./figs/dcls.png' width=800>
</div>

This is an official implementation of [Dilated Convolution with Learnable Spacings](https://arxiv.org/abs/2112.03740v4) by Ismail Khalfaoui Hassani, Thomas Pellegrini and Timothée Masquelier.

Dilated Convolution with Learnable Spacings (abbreviated to DCLS) is a novel convolution method based on gradient descent and interpolation. It could be seen as an improvement of the well known dilated convolution that has been widely explored in deep convolutional neural networks and which aims to inflate the convolutional kernel by inserting spaces between the kernel elements. 

In DCLS, the positions of the weights within the convolutional kernel are learned in a gradient-based manner, and the inherent problem of non-differentiability due to the integer nature of the positions in the kernel is solved by taking advantage of an interpolation method. 

For now, the code has only been implemented on [PyTorch](https://pytorch.org/), using Pytorch. 

- [What's new](#whats-new)
- [Installation](#installation)
- [Usage](#usage)
- [Device Supports](#device-supports)
- [Publications and Citation](#publications-and-citation)
- [Contribution](#contribution)

The method is described in the article [Dilated Convolution with Learnable Spacings](https://arxiv.org/abs/2112.03740v4). The Gaussian and triangle versions are described in the arXiv preprint [Dilated Convolution with Learnable Spacings: beyond bilinear interpolation](https://arxiv.org/abs/2306.00817v2).

## What's new
**Sep 28, 2023**:
- 🚀 🚀 A new repository for audio classification on AudioSet using DCLS and with state-of-the-art vision models adapted to audio spectrograms. Please checkout the git repo 
[DCLS Audio](https://github.com/K-H-Ismail/DCLS-Audio) and / or the paper [Audio classification with Dilated Convolution with Learnable Spacings](https://arxiv.org/abs/2309.13972) [![arXiv](https://img.shields.io/badge/arXiv-2309.13972-b31b1b.svg?style=plastic)](https://arxiv.org/abs/2309.13972). Models checkpoints are available !

**Sep 22, 2023**:
- 🎉 🎉 The paper on DCLS Gaussian interpolation [Dilated Convolution with Learnable Spacings: beyond bilinear interpolation](https://arxiv.org/abs/2306.00817v2) [![arXiv](https://img.shields.io/badge/arXiv-2306.00817v2-b31b1b.svg?style=plastic)](https://arxiv.org/abs/2306.00817v2) has been published at [the Differentiable Almost Everything Workshop of the 40th International Conference on Machine Learning](https://differentiable.xyz/) [[ICML2023]](https://icml.cc/Conferences/2023).

**Jun 16, 2023**:
-   A new tutorial on how to use DCLS in vision backbones is now available:
[DCLS Vision Tutorial](https://github.com/K-H-Ismail/Dilated-Convolution-with-Learnable-Spacings-PyTorch/blob/main/tutorials/DCLS_Vision_Tutorial.ipynb) [![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nr2zF-Zw8QRuKN68Mfdw5uCtznBa-UpM?usp=sharing). 
-   A short blog post which summarizes the DCLS method has been published in Medium: [What is Dilated Convolution with Learnable Spacings (DCLS) and how to use it ?](https://medium.com/@khalfaoui.ismail/dea93c490a82) [![medium](https://img.shields.io/badge/Medium-12100E?style=plastic)](https://medium.com/@khalfaoui.ismail/dea93c490a82).

**Jun 2, 2023**:
-   New DCLS version supports Gaussian and triangle interpolations in addition to previous bilinear interpolation. To use it, please do:
```
pip3 install --upgrade --force-reinstall dcls

```
  or recompile after a git update.
  
```python
import torch
from DCLS.construct.modules import  Dcls2d

# Dcls2d with Gaussian interpolation. available versions : ["gauss", "max", "v1", "v0"]
m = Dcls2d(96, 96, kernel_count=26, dilated_kernel_size=17, padding=8, groups=96, version="gauss")
input = torch.randn(20, 96, 50, 100)
output = m(input)
loss = output.sum()
loss.backward()
print(output, m.weight.grad, m.P.grad, m.SIG.grad)
```  
- Learning techniques for this method are described in [Dilated Convolution with Learnable Spacings: beyond bilinear interpolation](https://arxiv.org/abs/2306.00817).

**Apr 16, 2023**:
-   Fix an important bug in Dcls1d version. Please reinstall the pip wheel via 
```
pip3 install --upgrade --force-reinstall dcls
```
  or recompile after a git update.

**Jan 7, 2023**:
-   Important modification to ConstructKernel{1,2,3}d algorithm which allows to use less memory, this modification enables very large kernel counts. For example:
```python
from DCLS.construct.modules import  Dcls2d

m = Dcls2d(96, 96, kernel_count=2000, dilated_kernel_size=7, padding=3, groups=96).cuda() 
```
After installation of the new version 0.0.3 of DCLS, the use remains unchanged. 

**Nov 8, 2022**:
-   Previous branch main is moved to branch cuda, now in main branch we have fully native torch conv{1,2,3}d.

**Sep 27, 2022**:
-   Code release for ConvNeXt-dcls experiments. See [**ConvNeXt-dcls**](https://github.com/K-H-Ismail/ConvNeXt-dcls).

## Installation

DCLS is based on PyTorch and CUDA. Please make sure that you have installed all the requirements before you install DCLS.

**Requirements**:
-   Pytorch version torch>=1.6.0. See [**torch**](https://pytorch.org/).

**Preferred versions**:
```bash
pip3 install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

**Install the latest developing version from the source codes**:

From [GitHub](https://github.com/K-H-Ismail/Dilated-Convolution-with-Learnable-Spacings-PyTorch):
```bash
git clone https://github.com/K-H-Ismail/Dilated-Convolution-with-Learnable-Spacings-PyTorch.git
cd Dilated-Convolution-with-Learnable-Spacings-PyTorch
python3 -m pip install --upgrade pip
python3 -m build 
python3 -m pip install dist/dcls-0.0.5-py3-none-any.whl 

```

**Install the last stable version from** [**PyPI**](https://pypi.org/project/DCLS/):

```bash
pip3 install dcls
```

## Usage
Dcls methods could be easily used as a substitue of Pytorch's nn.Conv**n**d classical convolution method:

```python
import torch
from DCLS.construct.modules import  Dcls2d

# With square kernels, equal stride and dilation
m = Dcls2d(16, 33, kernel_count=3, dilated_kernel_size=7)
input = torch.randn(20, 16, 50, 100)
output = m(input)
loss = output.sum()
loss.backward()
print(output, m.weight.grad, m.P.grad)
```
A typical use is with the separable convolution

```python
import torch
from DCLS.construct.modules import  Dcls2d

m = Dcls2d(96, 96, kernel_count=34, dilated_kernel_size=17, padding=8, groups=96)
input = torch.randn(128, 96, 56, 56)
output = m(input)
loss = output.sum()
loss.backward()
print(output, m.weight.grad, m.P.grad)
```

Dcls with different dimensions 
```python
import torch
from DCLS.construct.modules import  Dcls1d 

# Will construct kernels of size 7x7 with 3 elements inside each kernel
m = Dcls1d(3, 16, kernel_count=3, dilated_kernel_size=7)
input = torch.rand(8, 3, 32)
output = m(input)
loss = output.sum()
loss.backward()
print(output, m.weight.grad, m.P.grad)
```

```python
import torch
from DCLS.construct.modules import  Dcls3d

m = Dcls3d(16, 33, kernel_count=10, dilated_kernel_size=(7,8,9))
input = torch.randn(20, 16, 50, 100, 30)
output = m(input)
loss = output.sum()
loss.backward()
print(output, m.weight.grad, m.P.grad)
```

**DepthWiseConv2dImplicitGEMM for 2D-DCLS**:

For 2D-DCLS, to install and enable the DepthWiseConv2dImplicitGEMM, please follow the instructions of [RepLKNet](https://github.com/DingXiaoH/RepLKNet-pytorch#use-our-efficient-large-kernel-convolution-with-pytorch). Otherwise, Pytorch's native Conv2D method will be used.


## Device Supports
DCLS supports CPU and Nvidia CUDA GPU devices now.

-   [x] Nvidia GPU
-   [x] CPU

Make sure to have your data and model on CUDA GPU.

## Publications and Citation

If you use DCLS in your work, please consider to cite it as follows:

```
@inproceedings{
hassani2023dilated,
title={Dilated convolution with learnable spacings},
author={Ismail Khalfaoui-Hassani and Thomas Pellegrini and Timoth{\'e}e Masquelier},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=Q3-1vRh3HOA}
}

```

If you use DCLS with Gaussian or triangle interpolations in your work, please consider to cite as well:

```
@inproceedings{
khalfaoui-hassani2023dilated,
title={Dilated Convolution with Learnable Spacings: beyond bilinear interpolation},
author={Ismail Khalfaoui-Hassani and Thomas Pellegrini and Timoth{\'e}e Masquelier},
booktitle={ICML 2023 Workshop on Differentiable Almost Everything: Differentiable Relaxations, Algorithms, Operators, and Simulators},
year={2023},
url={https://openreview.net/forum?id=j8FPBCltB9}
}
```

## Contribution

This project is open source, therefore all your contributions are welcomed, whether it's reporting issues, finding and fixing bugs, requesting new features, and sending pull requests ...


