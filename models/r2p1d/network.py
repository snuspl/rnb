"""Taken as is from https://github.com/irhumshafkat/R2Plus1D-PyTorch/blob/master/network.py with a slight change in the imports.
"""

import torch
import sys
import torch.nn as nn
from torch.nn.modules.utils import _triple

from models.r2p1d.module import SpatioTemporalConv


class SpatioTemporalResBlock(nn.Module):
    r"""Single block for the ResNet network. Uses SpatioTemporalConv in 
        the standard ResNet block layout (conv->batchnorm->ReLU->conv->batchnorm->sum->ReLU)
        
        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output produced by the block.
            kernel_size (int or tuple): Size of the convolving kernels.
            downsample (bool, optional): If ``True``, the output size is to be smaller than the input. Default: ``False``
        """
    def __init__(self, in_channels, out_channels, kernel_size, downsample=False):
        super(SpatioTemporalResBlock, self).__init__()
        
        # If downsample == True, the first conv of the layer has stride = 2 
        # to halve the residual output size, and the input x is passed 
        # through a seperate 1x1x1 conv with stride = 2 to also halve it.

        # no pooling layers are used inside ResNet
        self.downsample = downsample
        
        # to allow for SAME padding
        padding = kernel_size//2

        if self.downsample:
            # downsample with stride =2 the input x
            self.downsampleconv = SpatioTemporalConv(in_channels, out_channels, 1, stride=2)
            self.downsamplebn = nn.BatchNorm3d(out_channels)

            # downsample with stride = 2when producing the residual
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding, stride=2)
        else:
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding)

        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU()

        # standard conv->batchnorm->ReLU
        self.conv2 = SpatioTemporalConv(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.outrelu = nn.ReLU()

    def forward(self, x):
        res = self.relu1(self.bn1(self.conv1(x)))    
        res = self.bn2(self.conv2(res))

        if self.downsample:
            x = self.downsamplebn(self.downsampleconv(x))

        return self.outrelu(x + res)


class SpatioTemporalResLayer(nn.Module):
    r"""Forms a single layer of the ResNet network, with a number of repeating 
    blocks of same output size stacked on top of each other
        
        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output produced by the layer.
            kernel_size (int or tuple): Size of the convolving kernels.
            layer_size (int): Number of blocks to be stacked to form the layer
            block_type (Module, optional): Type of block that is to be used to form the layer. Default: SpatioTemporalResBlock. 
            downsample (bool, optional): If ``True``, the first block in layer will implement downsampling. Default: ``False``
        """

    def __init__(self, in_channels, out_channels, kernel_size, layer_size, block_type=SpatioTemporalResBlock, downsample=False):
        
        super(SpatioTemporalResLayer, self).__init__()

        # implement the first block
        self.block1 = block_type(in_channels, out_channels, kernel_size, downsample)

        # prepare module list to hold all (layer_size - 1) blocks
        self.blocks = nn.ModuleList([])
        for i in range(layer_size - 1):
            # all these blocks are identical, and have downsample = False by default
            self.blocks += [block_type(out_channels, out_channels, kernel_size)]

    def forward(self, x):
        x = self.block1(x)
        for block in self.blocks:
            x = block(x)

        return x


class R2Plus1DNet(nn.Module):
    r"""Forms the overall ResNet feature extractor by initializng 5 layers, with the number of blocks in 
    each layer set by layer_sizes, and by performing a global average pool at the end producing a 
    512-dimensional vector for each element in the batch.
        
        Args:
            layer_sizes (tuple): An iterable containing the number of blocks in each layer
            block_type (Module, optional): Type of block that is to be used to form the layers. Default: SpatioTemporalResBlock. 
        """
    def __init__(self, layer_sizes, block_type=SpatioTemporalResBlock):
        super(R2Plus1DNet, self).__init__()

        # first conv, with stride 1x2x2 and kernel size 3x7x7
        self.conv1 = SpatioTemporalConv(3, 64, [3, 7, 7], stride=[1, 2, 2], padding=[1, 3, 3])
        # output of conv2 is same size as of conv1, no downsampling needed. kernel_size 3x3x3
        self.conv2 = SpatioTemporalResLayer(64, 64, 3, layer_sizes[0], block_type=block_type)
        # each of the final three layers doubles num_channels, while performing downsampling 
        # inside the first block
        self.conv3 = SpatioTemporalResLayer(64, 128, 3, layer_sizes[1], block_type=block_type, downsample=True)
        self.conv4 = SpatioTemporalResLayer(128, 256, 3, layer_sizes[2], block_type=block_type, downsample=True)
        self.conv5 = SpatioTemporalResLayer(256, 512, 3, layer_sizes[3], block_type=block_type, downsample=True)

        # global average pooling of the output
        self.pool = nn.AdaptiveAvgPool3d(1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.pool(x)
        
        return x.view(-1, 512)

class R2Plus1DClassifier(nn.Module):
    r"""Forms a complete ResNet classifier producing vectors of size num_classes, by initializng 5 layers, 
    with the number of blocks in each layer set by layer_sizes, and by performing a global average pool
    at the end producing a 512-dimensional vector for each element in the batch, 
    and passing them through a Linear layer.
        
        Args:
            num_classes(int): Number of classes in the data
            layer_sizes (tuple): An iterable containing the number of blocks in each layer
            block_type (Module, optional): Type of block that is to be used to form the layers. Default: SpatioTemporalResBlock. 
        """
    def __init__(self, num_classes, layer_sizes, block_type=SpatioTemporalResBlock):
        super(R2Plus1DClassifier, self).__init__()

        self.res2plus1d = R2Plus1DNet(layer_sizes, block_type)
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.res2plus1d(x)
        x = self.linear(x) 

        return x


class R2Plus1DLayerNet(nn.Module):
    """Instantiate only the necessary layers of R(2+1)D model. 

    start_idx and end_idx are assumed to be 1-indexed and not 0-indexed.
    """
    def __init__(self, start_idx, end_idx, num_classes, layer_sizes, block_type=SpatioTemporalResBlock):
        super(R2Plus1DLayerNet, self).__init__()
        
        self.end_idx = end_idx
        layer_indices = [x for x in range(start_idx, end_idx+1)]
        self.layer_list = []
        for i in range(len(layer_indices)):
            if layer_indices[i] == 1:
                self.conv1 = SpatioTemporalConv(3, 64, [3, 7, 7], stride=[1, 2, 2], padding=[1, 3, 3])
                self.layer_list.append(self.conv1)
            elif layer_indices[i] == 2:
                self.conv2 = SpatioTemporalResLayer(64, 64, 3, layer_sizes[i], block_type=block_type)
                self.layer_list.append(self.conv2)
            elif layer_indices[i] == 3:
                self.conv3 = SpatioTemporalResLayer(64, 128, 3, layer_sizes[i], block_type=block_type, downsample=True)
                self.layer_list.append(self.conv3)
            elif layer_indices[i] == 4: 
                self.conv4 = SpatioTemporalResLayer(128, 256, 3, layer_sizes[i], block_type=block_type, downsample=True)
                self.layer_list.append(self.conv4)
            else: # layer == 5 
                self.conv5 = SpatioTemporalResLayer(256, 512, 3, layer_sizes[i], block_type=block_type, downsample=True)
                self.pool = nn.AdaptiveAvgPool3d(1)
                self.layer_list.extend([self.conv5, self.pool])
    
    def forward(self, x):
        for layer in self.layer_list:
            x = layer(x) 
        return x.view(-1, 512) if self.end_idx == 5 else x  

class R2Plus1DLayerWrapper(nn.Module):
    """A thin wrapper for R2Plus1DLayerNet with an addition of fc classification layer. 

    This wrapper can be used to reuse PyTorch weights trained on R2Plus1DClassifier.
    The classification layer will be used only if the last layer of the R(2+1)D model is included. 
    """
    def __init__(self, start_idx, end_idx, num_classes, layer_sizes, block_type=SpatioTemporalResBlock):
        super(R2Plus1DLayerWrapper, self).__init__()
        
        self.res2plus1d = R2Plus1DLayerNet(start_idx, end_idx, num_classes, layer_sizes, block_type)
        self.end_idx = end_idx
        if end_idx == 5:
          self.linear = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.res2plus1d(x)
        return self.linear(x) if self.end_idx == 5 else x


class R2Plus1DLayer12Net(nn.Module):
    """A shorter version of R2Plus1DNet, using only the first two layers."""
    def __init__(self, layer_size, block_type=SpatioTemporalResBlock):
        super(R2Plus1DLayer12Net, self).__init__()
        self.conv1 = SpatioTemporalConv(3, 64, [3, 7, 7], stride=[1, 2, 2], padding=[1, 3, 3])
        self.conv2 = SpatioTemporalResLayer(64, 64, 3, layer_size, block_type=block_type)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class R2Plus1DLayer12Wrapper(nn.Module):
    """A thin wrapper for R2Plus1DLayer12Net without any additional features.

    This wrapper is used solely for the purpose of reusing PyTorch weights
    trained on R2Plus1DClassifier with minimum adjustments.
    """
    def __init__(self, layer_size, block_type=SpatioTemporalResBlock):
        super(R2Plus1DLayer12Wrapper, self).__init__()
        self.res2plus1d = R2Plus1DLayer12Net(layer_size, block_type)

    def forward(self, x):
        return self.res2plus1d(x)


class R2Plus1DLayer345Net(nn.Module):
    """A shorter version of R2Plus1DNet, using only the last three layers."""
    def __init__(self, layer_sizes, block_type=SpatioTemporalResBlock):
        super(R2Plus1DLayer345Net, self).__init__()
        self.conv3 = SpatioTemporalResLayer(64, 128, 3, layer_sizes[0], block_type=block_type, downsample=True)
        self.conv4 = SpatioTemporalResLayer(128, 256, 3, layer_sizes[1], block_type=block_type, downsample=True)
        self.conv5 = SpatioTemporalResLayer(256, 512, 3, layer_sizes[2], block_type=block_type, downsample=True)
        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool(x)
        return x.view(-1, 512)


class R2Plus1DLayer345Wrapper(nn.Module):
    """Wrapper for R2Plus1DLayer345Net that adds a fc classification layer."""
    def __init__(self, num_classes, layer_sizes, block_type=SpatioTemporalResBlock):
        super(R2Plus1DLayer345Wrapper, self).__init__()
        self.res2plus1d = R2Plus1DLayer345Net(layer_sizes, block_type)
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.res2plus1d(x)
        x = self.linear(x) 
        return x
