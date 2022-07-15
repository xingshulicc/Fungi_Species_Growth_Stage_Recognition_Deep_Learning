# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Thu Mar 10 14:08:01 2022

@author: DELL
"""
import torch
from torch import nn
from torch.nn import init

class BasicModule(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels, 
                              kernel_size=3, 
                              stride=stride, 
                              padding=1, 
                              bias=False)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.act = nn.ReLU(inplace=True)
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False), 
                nn.BatchNorm2d(num_features=out_channels)
                )
        else:
            self.shortcut = nn.Identity()
        
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out += self.shortcut(x)
        out = self.act(out)
        
        return out

class Model_1(nn.Module):
    def __init__(self, block, num_blocks, classes):
        super().__init__()
        # set initial convolution block
        self.init_conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=2, padding=3, bias=False)
        self.init_bn = nn.BatchNorm2d(num_features=32)
        self.init_act = nn.ReLU(inplace=True)
        
        # stack building blocks
        self.block_1 = self._make_layer(block, 32, num_blocks[0])
        # the output shape of block_1 is: (B, C, 56, 56)
        self.block_2 = self._make_layer(block, 64, num_blocks[1])
        # the output shape of block_2 is: (B, C, 28, 28)
        self.block_3 = self._make_layer(block, 128, num_blocks[2])
        # the output shape of block_3 is: (B, C, 14, 14)
        self.block_4 = self._make_layer(block, 256, num_blocks[3])
        # the output shape of block_4 is: (B, C, 7, 7)
        
        # set classifier layer: fc_1 is species label prediction, fc_2 is time label prediction
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        # the output shape of gap is: (B, C, 1, 1)
        self.fc_1 = nn.Linear(in_features=512, out_features=classes[0])
        self.fc_2 = nn.Linear(in_features=512, out_features=classes[1])
    
    def _make_layer(self, block, in_channels, num_blocks):
        strides = [1] * (num_blocks-1) + [2]
        layers = []
        for stride in strides:
            out_channels = in_channels * stride
            layers.append(block(in_channels, out_channels, stride))
            
        return nn.Sequential(*layers)
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    
    def forward(self, x):
        out = self.init_conv(x)
        out = self.init_bn(out)
        out = self.init_act(out)
        
        out = self.block_1(out)
        out = self.block_2(out)
        out = self.block_3(out)
        out = self.block_4(out)
        
        b, c, _, _ = out.size()
        out = self.gap(out).view(b, c)
        label_1 = self.fc_1(out)
        label_2 = self.fc_2(out)
        
        return {'Species': label_1, 'Time': label_2}

def test_model():
    return Model_1(block=BasicModule, num_blocks=[2, 2, 2, 1], classes=[19, 4])


# if __name__  == "__main__":
#     modelviz = test_model()
#     sampledata = torch.rand(1, 3, 224, 224)
#     out = modelviz(sampledata)
#     print(out)


