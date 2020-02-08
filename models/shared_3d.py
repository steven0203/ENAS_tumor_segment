import numpy as np
from collections import defaultdict, deque

import torch 
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.shared_base import *
from utils import get_logger, get_variable, keydefaultdict

logger = get_logger()


def init_weight(idx , action , in_channels ,filters):
    if action=='3x3x3':
        return nn.Sequential(
            nn.Conv3d(in_channels,filters,kernel_size=3,stride=1,padding=1),
            nn.InstanceNorm3d(filters),
            nn.LeakyReLU(inplace=True)
        )
        
    elif action=='3x3x3 dilation 2':
        return nn.Sequential(
            nn.Conv3d(in_channels,filters,kernel_size=3,stride=1,padding=2,dilation=2),
            nn.InstanceNorm3d(filters),
            nn.LeakyReLU(inplace=True)
        )
    elif action=='3x3x3 dilation 3':
        return nn.Sequential(
            nn.Conv3d(in_channels,filters,kernel_size=3,stride=1,padding=3,dilation=3),
            nn.InstanceNorm3d(filters),
            nn.LeakyReLU(inplace=True)
        )
    elif action=='max pool':
        if idx == 0:
            return nn.Sequential(
                nn.MaxPool3d(kernel_size=3,stride=1,padding=1),
                nn.Conv3d(in_channels,filters,kernel_size=1,stride=1),
                nn.InstanceNorm3d(filters),
                nn.LeakyReLU(inplace=True)
            )
        return nn.MaxPool3d(kernel_size=3,stride=1,padding=1)
    elif action=='avg pool':
        if idx == 0:
            return nn.Sequential(
                nn.AvgPool3d(kernel_size=3,stride=1,padding=1),
                nn.Conv3d(in_channels,filters,kernel_size=1,stride=1),
                nn.InstanceNorm3d(filters),
                nn.LeakyReLU(inplace=True)
            )
        return nn.AvgPool3d(kernel_size=3,stride=1,padding=1)
    elif action=='identity' :
        if idx ==0:
            return nn.Sequential(
                nn.Conv3d(in_channels,filters,kernel_size=1,stride=1),
                nn.InstanceNorm3d(filters),
                nn.LeakyReLU(inplace=True)
            )
        return nn.Sequential()
    elif action=='sep 3x3x3':
        return nn.Sequential(
            nn.Conv3d(in_channels,in_channels,kernel_size=3,stride=1,padding=1,groups=in_channels),
            nn.Conv3d(in_channels,filters,kernel_size=1,stride=1),
            nn.InstanceNorm3d(filters),
            nn.LeakyReLU(inplace=True)
        )
        
class Unet(SharedModel):
    def __init__(self, args):
        super(Unet, self).__init__()

        self.args = args
        self.down_left_block = []
        self.down_right_block = []

        self.up_left_block = []
        self.up_right_block = []
        self._actions=[]
        in_channels = self.args.in_channels
        
        for layer in range(self.args.layers):
            self.down_left_block.append([])
            self.down_right_block.append([])
            self.up_left_block.append([])
            self.up_right_block.append([])
            filters =self.args.filters*(2**(layer))
            for block in range(self.args.num_blocks):
                self.down_left_block[layer].append([])
                self.down_right_block[layer].append([])
                self.up_left_block[layer].append([])
                self.up_right_block[layer].append([])
                for idx in range(block+1):
                    self.down_left_block[layer][block]={}
                    self.down_right_block[layer][block]={}
                    self.up_left_block[layer][block]={}
                    self.up_right_block[layer][block]={}
                    for action in self.args.shared_cnn_types:
                        if idx ==0:
                            down_in=in_channels
                            up_in=filters*self.args.num_blocks+filters*2*self.args.num_blocks
                        else:
                            up_in=filters
                            down_in=filters
                        self.down_left_block[layer][idx][action]=init_weight(idx,action,down_in,filters)
                        self.down_right_block[layer][idx][action]=init_weight(idx,action,down_in,filters)
                        self.up_left_block[layer][idx][action]=init_weight(idx,action,up_in,filters)
                        self.up_right_block[layer][idx][action]=init_weight(idx,action,up_in,filters)
                        self._actions.append(self.down_left_block[layer][idx][action])
                        self._actions.append(self.down_right_block[layer][idx][action])
                        self._actions.append(self.up_left_block[layer][idx][action])
                        self._actions.append(self.up_right_block[layer][idx][action])
            in_channels=filters*self.args.num_blocks
                    
        self.down_left_block.append([])
        self.down_right_block.append([])
        filters=self.args.filters*(2**(self.args.layers))
        for block in range(self.args.num_blocks):
            self.down_left_block[-1].append([])
            self.down_right_block[-1].append([])
            for idx in range(block+1):
                self.down_left_block[-1][block]={}
                self.down_right_block[-1][block]={}
                for action in self.args.shared_cnn_types:
                    if idx ==0:
                        down_in=in_channels
                    else:
                        down_in=filters
                    self.down_left_block[-1][idx][action] = init_weight(idx,action,down_in,filters)
                    self.down_right_block[-1][idx][action] = init_weight(idx,action,down_in,filters)
                    self._actions.append(self.down_left_block[-1][idx][action])
                    self._actions.append(self.down_right_block[-1][idx][action])
        self.out=nn.Conv3d(self.args.filters*self.args.num_blocks,self.args.n_classes,1)
        self._actions=nn.ModuleList(self._actions)
        self.reset_parameters()
        #self._actions=None
    def forward(self, inputs, dag):
        x=inputs
        outputs=[]
        for layer in range(self.args.layers+1):
            ip=[]
            ip.append(x)
            for b in range(self.args.num_blocks):
                left_idx,right_idx,left_action,right_action=self.get_actions(dag,layer,b)
                left_ip=ip[left_idx]
                right_ip=ip[right_idx]
                left_out=self.down_left_block[layer][left_idx][left_action](left_ip)
                right_out=self.down_right_block[layer][right_idx][right_action](right_ip)
                x=left_out+right_out
                ip.append(x)
            x = torch.cat(ip[1:], dim=1)
            outputs.append(x)
            x =nn.MaxPool3d(2)(x)
            

        x=outputs[-1]
        for layer in range(self.args.layers):
            ip=[]
            x=nn.Upsample(scale_factor=2,mode='trilinear',align_corners=True)(x)
            x=torch.cat([x,outputs[-layer-2]], dim=1) 
            ip.append(x)           
            for b in range(self.args.num_blocks):
                left_idx,right_idx,left_action,right_action=self.get_actions(dag,layer,b)
                left_ip=ip[left_idx]
                right_ip=ip[right_idx]
                left_out=self.up_left_block[-layer-1][left_idx][left_action](left_ip)
                right_out=self.up_right_block[-layer-1][right_idx][right_action](right_ip)
                x=left_out+right_out
                ip.append(x)
            x=torch.cat(ip[1:], dim=1)
        x=self.out(x)
        x=F.softmax(x,dim=1)
        return x

    def get_num_cell_parameters(self, dag):
        pass

    def reset_parameters(self):
        init_range = 0.025 
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)


    def set_arc(self,dag):
        actions=[]
        for layer in range(self.args.layers+1):
            for b in range(self.args.num_blocks):
                left_idx=dag[b*2][0]
                right_idx=dag[b*2+1][0]
                left_action=dag[b*2][1]
                right_action=dag[b*2+1][1]
                actions.append(self.down_left_block[layer][left_idx][left_action])
                actions.append(self.down_right_block[layer][right_idx][right_action])
            
        for layer in range(self.args.layers):
            for b in range(self.args.num_blocks):
                left_idx=dag[b*2][0]
                right_idx=dag[b*2+1][0]
                left_action=dag[b*2][1]
                right_action=dag[b*2+1][1]
                actions.append(self.up_left_block[-layer-1][left_idx][left_action])
                actions.append(self.up_right_block[-layer-1][right_idx][right_action])
        self._actions=nn.ModuleList(actions)
        
    def get_actions(self,dag,layer,block):
        if not self.args.multi_layer:
            left_idx=dag[block*2][0]
            right_idx=dag[block*2+1][0]
            left_action=dag[block*2][1]
            right_action=dag[block*2+1][1]
            return left_idx,right_idx,left_action,right_action
        else:
            left_idx=dag[layer*2*self.args.num_blocks+block*2][0]
            right_idx=dag[layer*2*self.args.num_blocks+block*2+1][0]
            left_action=dag[layer*2*self.args.num_blocks+block*2][1]
            right_action=dag[layer*2*self.args.num_blocks+block*2+1][1]
            return left_idx,right_idx,left_action,right_action
