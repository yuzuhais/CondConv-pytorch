# CondConv

Implementation of [CondConv: Conditionally Parameterized Convolutions for Efficient Inference](https://arxiv.org/abs/1904.04971) 
in PyTorch.

## Abstract

Convolutional layers are one of the basic building blocks of modern deep neural networks. One fundamental assumption is that convolutional kernels should
be shared for all examples in a dataset. We propose conditionally parameterized convolutions (CondConv), which learn specialized convolutional kernels
for each example. Replacing normal convolutions with CondConv enables us
to increase the size and capacity of a network, while maintaining efficient inference. We demonstrate that scaling networks with CondConv improves the
performance and inference cost trade-off of several existing convolutional neural
network architectures on both classification and detection tasks. On ImageNet
classification, our CondConv approach applied to EfficientNet-B0 achieves state-ofthe-art performance of 78.3% accuracy with only 413M multiply-adds. Code and
checkpoints for the CondConv Tensorflow layer and CondConv-EfficientNet models are available at: https://github.com/tensorflow/tpu/tree/master/
models/official/efficientnet/condconv.


## Installation

    pip install git+https://github.com/nibuiro/CondConv-pytorch.git

## Usage


For 2D inputs (CondConv2D):

```python
import torch
from condconv import CondConv2D


batch_size = 1 # You need update param a sample.


class Model(nn.Module):
    def __init__(self, num_experts):
        super(Model, self).__init__()
        self.condconv2d = CondConv2D(10, 128, kernel_size=1, num_experts=num_experts)
        
    def forward(self, x):
        x = self.condconv2d(x)
```

## Reference
[Yang et al., 2019] CondConv: Conditionally Parameterized Convolutions for Efficient Inference
