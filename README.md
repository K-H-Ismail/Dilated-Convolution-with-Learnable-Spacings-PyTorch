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
coming soon
```

**Install the latest developing version from the source codes**:

From [GitHub](https://github.com/K-H-Ismail/Dilated-Convolution-with-Learnable-Spacings-PyTorch):
```bash
git clone https://github.com/K-H-Ismail/Dilated-Convolution-with-Learnable-Spacings-PyTorch.git
cd Dilated-Convolution-with-Learnable-Spacings-PyTorch
python ./setup.py install 
```
To prevent bad install directory or ```PYTHONPATH```, please use
```bash 
export PYTHONPATH=path/to/your/Python-Ver/lib/pythonVer/site-packages/
python ./setup.py install --prefix=path/to/your/Python-Ver/
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
The constructive DCLS method presents a performance problem when moving to larger dilations (greater than 7). Indeed, the constructed kernel is largely sparse (it has a sparsity of 1 - 1/(d1 * d2)) and the zeros are effectively taken into account during the convolution leading to great losses of performance in time and memory and this all the more as the dilation is large.

This is why we implemented an alternative method by adapting the im2col algorithm  which aims to speed up the convolution by unrolling the input into a Toepliz matrix and then performing matrix multiplication.

You can use both methods by importing the suitable modules as follows:

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
DCLS-im2col doesn't support mixed precision operations for now. By default every tensor is converted to have float32 precision.

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


