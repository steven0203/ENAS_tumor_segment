import numpy as np
import torch 
from torch import nn
import torch.nn.functional as F



class Unet3D(torch.nn.Module):
    def __init__(self,in_channels=4,labels=5,base_filters=30,depth=5):
        super(Unet3D, self).__init__()
        self.in_channels=in_channels
        self.labels=labels
        self.base_filters=base_filters
        self.depth=depth
        self.down_conv_block=[]
        self.up_conv_block=[]

        channels=in_channels
        filters=base_filters
        for _ in range(depth):
            self.down_conv_block.append(create_conv_block(channels,filters))
            channels=filters
            self.down_conv_block.append(create_conv_block(channels,filters))
            filters=filters*2
        
        filters=filters//4
        channels=channels+filters
        for _ in range(depth-1):
            self.up_conv_block.append(create_conv_block(channels,filters))
            channels=filters
            self.up_conv_block.append(create_conv_block(channels,filters))
            filters=filters//2
            channels=channels+filters

        self.out=nn.Conv3d(channels,self.labels,1)
        self._actions=nn.ModuleList(self.down_conv_block+self.up_conv_block)
        self.reset_parameters()

    def forward(self,inputs):
        x=inputs
        down=[]
        for i in range(self.depth):
            x=self.down_conv_block[i*2](x)
            x=self.down_conv_block[i*2+1](x)
            down.append(x)
            x=nn.MaxPool3d(2)(x)
        
        x=down[-1]
        for i in range(self.depth-1):
            x=nn.Upsample(scale_factor=2,mode='trilinear',align_corners=True)(x)
            x=torch.cat([down[-i-2],x],dim=1)
            x=self.up_conv_block[i*2](x)
            x=self.up_conv_block[i*2+1](x)
        
        x=self.out(x)
        x=F.softmax(x,dim=1)
        return x

    def reset_parameters(self):
        init_range = 0.025 
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)






def create_conv_block(in_channels,n_filters,kernel=(3,3,3),strides=(1,1,1),padding=1):
    return nn.Sequential(
        nn.Conv3d(in_channels,n_filters,kernel_size=kernel,stride=strides,padding=padding),
        nn.InstanceNorm3d(n_filters),
        nn.LeakyReLU(inplace=True)
    )