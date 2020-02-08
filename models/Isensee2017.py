import numpy as np
import torch 
from torch import nn



class isensee2017_model(torch.nn.Module):
    def __init__(self,in_channels=4,labels=4,base_filters=16,depth=5,n_segmentation_levels=3):
        super(isensee2017_model,self).__init__()
        self.in_channels=in_channels
        self.labels=labels
        self.base_filters=base_filters
        self.depth=depth
        self.n_segmentation_levels=n_segmentation_levels
        self.conv_block=[]
        self.context_module=[]
        self.upsampe_module=[]
        self.local_module=[]
        self.segment_layers=[]
        for i in range(depth):
            if i==0:
                self.conv_block.append(create_conv_block(in_channels,base_filters))
            else:
                self.conv_block.append(create_conv_block(2**(i-1)*base_filters,2**i*base_filters,strides=(2,2,2)))
                self.upsampe_module.append(create_upsample_module(2**(depth-i)*base_filters,2**(depth-i-1)*base_filters))
            self.context_module.append(create_context_module(2**i*base_filters,2**i*base_filters))
            
        for i in range(depth-2):
            self.local_module.append(create_localization_module(2**(depth-i-1)*base_filters,2**(depth-i-2)*base_filters))
        
        self.final_conv_block=create_conv_block(2*base_filters,2*base_filters)
        for i in range(n_segmentation_levels-1):
            self.segment_layers.append(nn.Conv3d(2**(n_segmentation_levels-i-1)*base_filters,labels,kernel_size=(1,1,1)))
        self.segment_layers.append(nn.Conv3d(2*base_filters,labels,kernel_size=(1,1,1)))
        self._actions=self.conv_block+self.context_module+self.upsampe_module+self.local_module+self.segment_layers
        self._actions.append(self.final_conv_block)
        self._actions=nn.ModuleList(self._actions)
        self.reset_parameters()


    def forward(self,inputs):
        x=inputs
        down=[]
        for i in range(self.depth):
            x=self.conv_block[i](x)
            tmp=x
            x=self.context_module[i](x)
            x=tmp+x
            if i !=self.depth-1:
                down.append(x)
        x=self.upsampe_module[0](x)

        up_out=[]
        for i in range(self.depth-2):
            up_in=torch.cat([down[-i-1],x],dim=1)
            x=self.local_module[i](up_in)
            up_out.append(x)
            x=self.upsampe_module[i+1](x)
        up_in=torch.cat([down[0],x],dim=1)
        x=self.final_conv_block(up_in)
        up_out.append(x)
        x=self.segment_layers[0](up_out[-self.n_segmentation_levels])
        tmp=nn.Upsample(scale_factor=2,mode='trilinear',align_corners=True)(x)
        for i in range(self.n_segmentation_levels-1):
            x=self.segment_layers[i+1](up_out[-self.n_segmentation_levels+i+1])
            x=x+tmp
            if i !=self.n_segmentation_levels-2:
                tmp=nn.Upsample(scale_factor=2,mode='trilinear',align_corners=True)(x)
        
        return nn.Softmax(dim=1)(x)

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

def create_context_module(in_channels,n_filters):
    return nn.Sequential(
        create_conv_block(in_channels,n_filters),
        nn.Dropout3d(p=0.3),
        create_conv_block(in_channels,n_filters)
    )

def create_upsample_module(in_channels,n_filters):
    return nn.Sequential(
        nn.Upsample(scale_factor=2,mode='trilinear',align_corners=True),
        create_conv_block(in_channels,n_filters)
    )

def create_localization_module(in_channels,n_filters):
    return nn.Sequential(
        create_conv_block(in_channels,n_filters),
        create_conv_block(n_filters,n_filters,kernel=(1,1,1),padding=0)
    )
