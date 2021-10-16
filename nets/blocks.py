'''
Created on Jun 10, 2021

@author: Quang TRAN
'''

from torch import nn


class ResidualBlock(nn.Module):

    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.relu(x)
        out = self.bn(out)
        out = self.conv(out)
        out = self.relu1(out)
        out = self.bn1(out)
        out = self.conv1(out)
        out += residual
        
        return out


class DenseBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False):
        super(DenseBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.activation = nn.LeakyReLU(out_channels)
        
    def forward(self, x):
        out = self.conv(x)
        out = self.activation(out)
        
        return out
