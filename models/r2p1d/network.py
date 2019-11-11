import torch.nn as nn

r2p1d_module =  __import__('R2Plus1D-PyTorch.module', fromlist=('SpatioTemporalConv'))
SpatioTemporalConv = r2p1d_module.SpatioTemporalConv
r2p1d_network =  __import__('R2Plus1D-PyTorch.network', fromlist=('SpatioTemporalResBlock, SpatioTemporalResLayer'))
SpatioTemporalResBlock = r2p1d_network.SpatioTemporalResBlock
SpatioTemporalResLayer = r2p1d_network.SpatioTemporalResLayer

class R2Plus1DLayerNet(nn.Module):
    """Instantiate only the necessary layers of R(2+1)D model. 

    start_idx and end_idx are assumed to be 1-indexed and not 0-indexed.
    """
    def __init__(self, start_idx, end_idx, num_classes, layer_sizes, block_type=SpatioTemporalResBlock):
        super(R2Plus1DLayerNet, self).__init__()
        
        self.end_idx = end_idx
        layer_indices = range(start_idx, end_idx+1)
        self.layer_list = []
        for i, layer_index in enumerate(layer_indices):
            if layer_index == 1:
                self.conv1 = SpatioTemporalConv(3, 64, [3, 7, 7], stride=[1, 2, 2], padding=[1, 3, 3])
                self.layer_list.append(self.conv1)
            elif layer_index == 2:
                self.conv2 = SpatioTemporalResLayer(64, 64, 3, layer_sizes[i], block_type=block_type)
                self.layer_list.append(self.conv2)
            elif layer_index == 3:
                self.conv3 = SpatioTemporalResLayer(64, 128, 3, layer_sizes[i], block_type=block_type, downsample=True)
                self.layer_list.append(self.conv3)
            elif layer_index == 4: 
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
