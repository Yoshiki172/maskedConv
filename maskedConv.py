import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import torchvision

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import time


class MaskedConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):
        super(MaskedConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.biasbool = bias
        self.padding_mode = padding_mode
        self.device = device
        self.dtype = dtype
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.normalConv = torch.nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.biasbool, self.padding_mode, self.device, self.dtype).to(self.device)
        self.weight = self.normalConv.weight.data
        
        if self.biasbool == True:
            self.bias = self.normalConv.bias
            
        else:
            self.bias = torch.zeros(self.out_channels).to(self.device)
            
        
        self.UnFold = torch.nn.Unfold(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation).to(self.device)
    #@profile
    def forward(self, x, mask):
        input = x.clone().detach()
        batch, in_channels, in_height, in_width = x.size()
        _, _, kernel_height, kernel_width = self.weight.size()
        
        x,mask = x.to(self.device),mask.to(self.device)
        x = self.UnFold(torch.cat([x,mask],dim=1))
        mask = x[:,kernel_height*kernel_width*self.in_channels:kernel_height*kernel_width*(self.in_channels+1),:].transpose_(1,2)
        x = x[:,0:kernel_height*kernel_width*self.in_channels,:].transpose_(1,2)

        mask = torch.any(mask != 0, dim=2)
        summask = torch.sum(mask, dim=1)

        x = x[mask].unsqueeze(dim=0).permute(0,2,1)
        
        weight_view = self.weight.view((self.out_channels,-1))
        bias_view = self.bias.view(1, self.bias.shape[0], 1).to(self.device)
        x = torch.matmul(weight_view, x) + bias_view
        
        mask = mask.unsqueeze(dim=1)
        mask = mask.repeat(1, self.out_channels, mask.shape[1])
        
        output = torch.zeros((batch,self.out_channels,mask.size(2)))
        start = 0

        for i in range(batch):
            end = start + summask[i].item()
            output[i].masked_scatter_(mask[i],x[0:1,0:self.out_channels,start:end].squeeze(0))
            start = end
        

        out_height = (in_height + 2 * self.padding - kernel_height) // self.stride + 1
        out_width = (in_width + 2 * self.padding - kernel_width) // self.stride + 1
        output = output.reshape(batch,self.out_channels,out_height,out_width)
        
        return output
    
