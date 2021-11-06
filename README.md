# Dilated-Convolution-with-Learnable-Spacings-PyTorch
Ismail Khalfaoui Hassani

Dilated Convolution with Learnable Spacings (abbreviated to DCLS) is a novel convolution method based on gradient descent and interpolation. It could be seen as an improvement of the well known dilated convolution that has been widely explored in deep convolutional neural networks and which aims to inflate the convolutional kernel by inserting spaces between the kernel elements. 

In DCLS, the positions of the weights within the convolutional kernel are learned in a gradient-based manner, and the inherent problem of non-differentiability due to the integer nature of the positions in the kernel is solved by taking advantage of an interpolation method. 

For now, the code has only been implemented on [PyTorch](https://pytorch.org/), using Pytorch's C++ API and custom cuda extensions. 

- [Installation](#installation)
- [Usage](#usage)
- [Device Supports](#device-supports)
- [Publications and Citation](#publications-and-citation)
- [Contribution](#contribution)

## Installation

DCLS is based on PyTorch and CUDA. Please make sure that you have installed all the requirements before you install DCLS.


**Install the last stable version from** [**PyPI**](https://pypi.org/project/DCLS/):

```bash
pip install DCLS
```

**Install the latest developing version from the source codes**:

From [GitHub](https://github.com/K-H-Ismail/Dilated-Convolution-with-Learnable-Spacings-PyTorch):
```bash
git clone https://github.com/K-H-Ismail/Dilated-Convolution-with-Learnable-Spacings-PyTorch.git
cd Dilated-Convolution-with-Learnable-Spacings-PyTorch
python ./setup.py install 
```

## Usage
Dcls methods could be easily used as a substitue of Pytorch's nn.Conv**n**d classical convolution method:

```python
from DCLS.modules.Dcls import Dcls2d

# With square kernels, equal stride and dilation
m = Dcls2d(16, 33, 3, dilation=4, stride=2)
# non-square kernels and unequal stride and with padding`and dilation
m = Dcls2d(16, 33, (3, 5), dilation=4, stride=(2, 1), padding=(4, 2))
# non-square kernels and unequal stride and with padding and dilation
m = Dcls2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 2))
# non-square kernels and unequal stride and with padding and dilation
m = Dcls2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 2))
# With square kernels, equal stride, dilation and a scaling gain for the positions
m = Dcls2d(16, 33, 3, dilation=4, stride=2, gain=10)
input = torch.randn(20, 16, 50, 100)
output = m(input)

```
__**Note:**__ using ```Dcls2d``` with a ```dilation``` argument of 1 basically amounts to using ```nn.Conv2d```, therfore DCLS should always be used with a dilation > 1.

## Construct and Im2col methods

```python
from DCLS.construct.modules.Dcls import  Dcls2d as cDcls2d

# Will construct three (33, 16, (3x4), (3x4)) Tensors for weight, P_h positions and P_w positions 
m = cDcls2d(16, 33, 3, dilation=4, stride=2, gain=10)
input = torch.randn(20, 16, 50, 100)
output = m(input)

```

```python
from DCLS.modules.Dcls import  Dcls2d 

# Will not construct kernels and will perform im2col algorithm instead 
m = Dcls2d(16, 33, 3, dilation=4, stride=2, gain=10)
input = torch.randn(20, 16, 50, 100)
output = m(input)

```
__**Note:**__ in the im2col Dcls method the two extra learnable parameters P_h and P_w are of size ```channels_in // group x kernel_h x kernel_w```, while in the construct method they are of size ```channels_out x channels_in // group x kernel_h x kernel_w```

## Device Supports
DCLS only supports Nvidia CUDA GPU devices for the moment. The CPU version has not been implemented yet.

-   [x] Nvidia GPU
-   [ ] CPU

Make sure to have your data and model on CUDA GPU.

## Publications and Citation

If you use DCLS in your work, please consider to cite it as follows:

```
@misc{Dilated Convolution with Learnable Spacings,
	title = {Dilated Convolution with Learnable Spacings},
	author = {Ismail Khalfaoui Hassani},
	year = {2021},
	howpublished = {\url{https://github.com/K-H-Ismail/Dilated-Convolution-with-Learnable-Spacings-PyTorch}},
	note = {Accessed: YYYY-MM-DD},
}
```

## Contribution

This project is open source, therefore all your contributions are welcomed, whether it's reporting issues, finding and fixing bugs, requesting new features, and sending pull requests ...


