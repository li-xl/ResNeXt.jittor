#coding=utf-8
import jittor as jt 
import numpy as np 
from jittor import Module,nn,init

'''
convert from https://github.com/prlz77/ResNeXt.pytorch
'''

class ResNeXtBottleneck(Module):
    """
    RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """

    def __init__(self, in_channels, out_channels, stride, cardinality, base_width, widen_factor):
        """ Constructor

        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            cardinality: num of convolution groups.
            base_width: base number of channels in each group.
            widen_factor: factor to reduce the input dimensionality before convolution.
        """
        super(ResNeXtBottleneck, self).__init__()
        width_ratio = out_channels / (widen_factor * 64.)
        D = cardinality * int(base_width * width_ratio)
        self.conv_reduce = nn.Conv(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm(D)
        self.conv_conv = nn.Conv(D, D, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn = nn.BatchNorm(D)
        self.conv_expand = nn.Conv(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm(out_channels)
        self.relu = nn.Relu()

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.append(nn.Conv(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,bias=False))
            self.shortcut.append(nn.BatchNorm(out_channels))

    def execute(self, x):
        bottleneck = self.conv_reduce(x)
        bottleneck = self.relu(self.bn_reduce(bottleneck))
        bottleneck = self.conv_conv(bottleneck)
        bottleneck = self.relu(self.bn(bottleneck))
        bottleneck = self.conv_expand(bottleneck)
        bottleneck = self.bn_expand(bottleneck)
        residual = self.shortcut(x)
        x = self.relu(residual + bottleneck)
        return x


class CifarResNeXt(Module):
    """
    ResNext optimized for the Cifar dataset, as specified in
    https://arxiv.org/pdf/1611.05431.pdf
    """

    def __init__(self, cardinality, depth, nlabels, base_width, widen_factor=4):
        """ Constructor

        Args:
            cardinality: number of convolution groups.
            depth: number of layers.
            nlabels: number of classes
            base_width: base number of channels in each group.
            widen_factor: factor to adjust the channel dimensionality
        """
        super(CifarResNeXt, self).__init__()
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth - 2) // 9
        self.base_width = base_width
        self.widen_factor = widen_factor
        self.nlabels = nlabels
        self.output_size = 64
        self.stages = [64, 64 * self.widen_factor, 128 * self.widen_factor, 256 * self.widen_factor]

        self.conv_1_3x3 = nn.Conv(3, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm(64)
        self.stage_1 = self.block('stage_1', self.stages[0], self.stages[1], 1)
        self.stage_2 = self.block('stage_2', self.stages[1], self.stages[2], 2)
        self.stage_3 = self.block('stage_3', self.stages[2], self.stages[3], 2)
        self.classifier = nn.Linear(self.stages[3], nlabels)

        self.pool = nn.Pool(8,1,op="mean")

        self.relu = nn.Relu()
        init.relu_invariant_gauss_(self.classifier.weight)

        for param in self.parameters():
            key = param.name()
            if key.split('.')[-1] == 'weight':
                if 'Conv' in key:
                    init.relu_invariant_gauss_(param, mode='fan_out')
                if 'BatchNorm' in key:
                    init.constant_(param, value=1.0)
            elif key.split('.')[-1] == 'bias':
                init.constant_(param, value=0.0)

    def block(self, name, in_channels, out_channels, pool_stride=2):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.

        Args:
            name: string name of the current block.
            in_channels: number of input channels
            out_channels: number of output channels
            pool_stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.

        Returns: a Module consisting of n sequential bottlenecks.

        """
        block = nn.Sequential()
        for bottleneck in range(self.block_depth):
            if bottleneck == 0:
                block.append(ResNeXtBottleneck(in_channels, out_channels, pool_stride, self.cardinality,self.base_width, self.widen_factor))
            else:
                block.append(ResNeXtBottleneck(out_channels, out_channels, 1, self.cardinality, self.base_width,self.widen_factor))
        return block

    def execute(self, x):
        x = self.conv_1_3x3(x)
        x = self.relu(self.bn_1(x))
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.pool(x)
        x = x.reshape((-1, self.stages[3]))
        return self.classifier(x)
