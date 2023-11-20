from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math


# ---------------------- 2D Convolutions -------------------- #
# class name should be defined as a CamelCase
# Last update Sept 18, 2020
# ------------------------------------------------------------ #
class ConvBlock (nn.Module):
    def __init__(self, ninput, noutput,
                 kernel=3, stride=2, num_groups=32,
                 norm=None, activate=None, bias=True, inplace=False):
        super().__init__()

        if kernel % 2 == 0:
            self.padding = int((kernel/2) - 1)
        else:
            self.padding = int(kernel/2)

        self.conv = nn.Conv2d(ninput, noutput, kernel_size=kernel, stride=stride, padding=self.padding, bias=bias)

        if norm == 'batch':
            self.norm = nn.BatchNorm2d(noutput, eps=1e-3)
        elif norm == 'group':  # 32 groups by default
            self.norm = nn.GroupNorm(num_groups, noutput)
        elif norm == 'instance':
            self.norm = nn.InstanceNorm2d(noutput)  # # of groups is same as group
        elif norm == 'layer':
            self.norm = nn.GroupNorm(1, noutput)  # # of groups is same as group
        else:
            self.norm = None

        if activate == 'relu':
            self.activate = nn.ReLU(inplace=inplace)  # remove the input (do not use for ResBlock)
        elif activate == 'leaky':
            self.activate = nn.LeakyReLU(0.2, inplace=inplace)
        elif activate == 'tanh':
            self.activate = nn.Tanh()
        else:
            self.activate = None

    def forward(self, input):
        output = self.conv(input)
        if self.norm is not None:
            output = self.norm(output)
        if self.activate is not None:
            output = self.activate(output)
        return output


# upsample, when the stride is greater than 1.
class ConvTransBlock (nn.Module):
    def __init__(self, ninput, noutput,
                 kernel=3, stride=2, output_padding=1, num_groups=32,
                 norm=None, activate=None, slope=0.2, bias=True, inplace=True):
        super().__init__()
        if kernel % 2 == 0:
            self.padding = int ((kernel / 2) - 1)
            output_padding = 0
        else:
            self.padding = int (kernel / 2)
        self.conv = nn.ConvTranspose2d (ninput, noutput, kernel_size=kernel, stride=stride,
                                        padding=self.padding, output_padding=output_padding, bias=bias)

        if norm == 'batch':
            self.norm = nn.BatchNorm2d(noutput, eps=1e-3)
        elif norm == 'group':  # 32 groups by default
            self.norm = nn.GroupNorm(num_groups, noutput)
        elif norm == 'instance':
            self.norm = nn.InstanceNorm2d(noutput)  # # of groups is same as group
        elif norm == 'layer':
            self.norm = nn.GroupNorm(1, noutput)  # # of groups is same as group
        else:
            self.norm = None

        if activate == 'relu':
            self.activate = nn.ReLU(inplace=True)  # remove the input (do not use for ResBlock)
        elif activate == 'leaky':
            self.activate = nn.LeakyReLU(slope, inplace=True)
        else:
            self.activate = None

    def forward(self, input):
        output = self.conv(input)
        if self.norm is not None:
            output = self.norm(output)
        if self.activate is not None:
            output = self.activate(output)
        return output


# Initialization Block for ResNet-C (7x7 is replaced with three 3x3 blocks, 7x7 = 3x3x5.56 in terms of complexity)
class ConvInit_ResC (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv1 = nn.Conv2d (ninput, noutput, (3, 3), stride=2, padding=1, bias=True)
        self.conv2 = nn.Conv2d (ninput, noutput, (3, 3), stride=1, padding=1, bias=True)
        self.pool = nn.MaxPool2d (2, stride=2)
        self.bn = nn.BatchNorm2d (noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv2(output)
        output = self.pool(output)
        return self.bn(output)


# four submodules: DoubleConv, Up, Down, OutConv
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, inplace=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=inplace),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=inplace)
        )

    def forward(self, x):
        return self.double_conv(x)



# ResNetBlock with full pre-activation (The size of input channels does not change!)
# cannot change the size of input because of identity mapping!!!
# otherwise, down/upsampling is necessary for the input.
class ResBlockF (nn.Module):
    def __init__(self, ninput, noutput, kernel=3, stride=2, bias=True, downsample=False):
        super().__init__()
        self.inplanes = ninput
        self.outplanes = noutput
        self.bn1 = nn.BatchNorm2d (self.inplanes, eps=1e-3)
        self.conv1 = nn.Conv2d(self.inplanes, self.outplanes, kernel, stride=stride, padding=int(kernel/2), bias=bias)
        self.bn2 = nn.BatchNorm2d (self.outplanes, eps=1e-3)
        self.conv2 = nn.Conv2d (self.outplanes, self.outplanes, kernel, stride=stride, padding=int (kernel / 2), bias=bias)
        self.downsample = downsample

        if stride == 1:
            self.downsample = None
        elif stride == 2:
            self.downsample = nn.Sequential(
                        nn.Conv2d(self.inplanes, self.outplanes, kernel=1, stride=stride),
                        nn.BatchNorm2d (self.outplanes, eps=1e-3),
                    )

    def forward(self, input):
        identity = input
        output = self.bn(input)
        output = F.relu(output)
        output = self.conv(output)
        output = self.bn(output)
        output = F.relu (output)
        output = self.conv (output)
        if self.downsample is True:
            identity = self.downsample(input)
        output += identity
        return output


# CVPR 2019, Bag of Tricks for Image Classification with CNNs
# Downsampling function for ResNet-D
class DownSample_ResD (nn.Module):
    def __init__(self, ninput, noutput):  # originally ninput = 512, noutput = 2048
        super().__init__()

        self.conv_3x3 = nn.Conv2d(ninput, ninput, (3, 3), stride=2, padding=1, bias=True)
        self.conv_1x1_in = nn.Conv2d(ninput, ninput, (1, 1), stride=1, padding=0, bias=True)
        self.conv_1x1_out = nn.Conv2d(ninput, noutput, (1, 1), stride=1, padding=0, bias=True)
        self.pool = nn.AvgPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output1 = self.pool(input)
        output1 = self.conv_1x1_in(output1)

        output2 = self.conv_1x1_in(input)
        output2 = self.conv_3x3(output2)
        output2 = self.conv_1x1_out(output2)
        # concat or add?
        output = self.bn(output1 + output2)
        output += input
        return output


class ConvConcat3x3 (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput-ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)


class ConvRes3x3 (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput, (3, 3), stride=2, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output + input)


class NonBottleneck_1D (nn.Module):
    def __init__(self, ninput, dropprob=0.1, dilated=1):
        super ().__init__ ()

        self.conv3x1_1 = nn.Conv2d (ninput, ninput, (3, 1), stride=1, padding=(1, 0), bias=True)
        self.conv1x3_1 = nn.Conv2d (ninput, ninput, (1, 3), stride=1, padding=(0, 1), bias=True)
        self.bn1 = nn.BatchNorm2d (ninput, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d (ninput, ninput, (3, 1), stride=1, padding=(1 * dilated, 0), bias=True,
                                    dilation=(dilated, 1))
        self.conv1x3_2 = nn.Conv2d (ninput, ninput, (1, 3), stride=1, padding=(0, 1 * dilated), bias=True,
                                    dilation=(1, dilated))
        self.bn2 = nn.BatchNorm2d (ninput, eps=1e-03)

        self.dropout = nn.Dropout2d (dropprob)

    def forward(self, input):
        output = self.conv3x1_1 (input)
        output = F.relu (output)
        output = self.conv1x3_1 (output)
        output = self.bn1 (output)
        output = F.relu (output)

        output = self.conv3x1_2 (output)
        output = F.relu (output)
        output = self.conv1x3_2 (output)
        output = self.bn2 (output)

        if self.dropout.p != 0:
            output = self.dropout (output)

        return F.relu (output + input)


# ------------------------- 3D Convolutions -------------------- #
class conv3_basic (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.Conv3d(ninput, noutput, (3, 3), stride=2, padding=1, bias=True)
        self.bn = nn.BatchNorm3d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)


class ConvBlock3D (nn.Module):
    def __init__(self, ninput, noutput,
                 kernel=3, stride=2, num_groups=32,
                 norm=None, activate=None, bias=True, inplace=True):
        super().__init__()
        if kernel % 2 == 0:
            self.padding = int ((kernel / 2) - 1)
        else:
            self.padding = int (kernel / 2)

        self.conv = nn.Conv3d(ninput, noutput, kernel_size=kernel, stride=stride, padding=self.padding, bias=bias)

        if norm == 'batch':
            self.norm = nn.BatchNorm3d(noutput, eps=1e-3)
        elif norm == 'group':  # 32 groups by default
            self.norm = nn.GroupNorm(num_groups, noutput)
        elif norm == 'instance':
            self.norm = nn.GroupNorm(noutput, noutput)  # # of groups is same as group
        elif norm == 'layer':
            self.norm = nn.GroupNorm(1, noutput)  # # of groups is same as group
        else:
            self.norm = None

        if activate == 'relu':
            self.activate = nn.ReLU(inplace=inplace)  # remove the input (do not use for ResBlock)
        elif activate == 'leaky':
            self.activate = nn.LeakyReLU(inplace=inplace)
        else:
            self.activate = None

    def forward(self, input):
        output = self.conv(input)
        if self.norm is not None:
            output = self.norm(output)
        if self.activate is not None:
            output = self.activate(output)
        return output


class ConvTransBlock3D (nn.Module):
    def __init__(self, ninput, noutput,
                 kernel=3, stride=2, output_padding=1, num_groups=8,
                 norm=None, activate=None, bias=True, inplace=True):
        super().__init__()
        if kernel % 2 == 0:
            self.padding = int ((kernel / 2) - 1)
        else:
            self.padding = int (kernel / 2)
        self.conv = nn.ConvTranspose3d (ninput, noutput, kernel_size=kernel, stride=stride,
                                        padding=self.padding, output_padding=output_padding, bias=bias)

        if norm == 'batch':
            self.norm = nn.BatchNorm3d(noutput, eps=1e-3)
        elif norm == 'group':  # 32 groups by default
            self.norm = nn.GroupNorm(num_groups, noutput)
        elif norm == 'instance':
            self.norm = nn.GroupNorm(noutput, noutput)  # # of groups is same as group
        elif norm == 'layer':
            self.norm = nn.GroupNorm(1, noutput)  # # of groups is same as group
        else:
            self.norm = None

        if activate == 'relu':
            self.activate = nn.ReLU(inplace=inplace)  # remove the input (do not use for ResBlock)
        elif activate == 'leaky':
            self.activate = nn.LeakyReLU(inplace=inplace)
        else:
            self.activate = None

    def forward(self, input):
        output = self.conv(input)
        if self.norm is not None:
            output = self.norm(output)
        if self.activate is not None:
            output = self.activate(output)
        return output

if __name__ == '__main__':
    # a = Variable (torch.randn (10, 5)).float()
    # b = Variable (torch.randn (10, 5)).float ()
    # print(a*b)
    # loss_fn = torch.nn.L1Loss().cuda()
    input = Variable (torch.randn (4, 32, 64, 64)).float().cuda ()  # B x C x W x H
    target = Variable (torch.randn (4, 32, 64, 64)).float().cuda ()
    #
    model = ResBlockF(32, 32, kernel=3, stride=1).cuda()
    output = model(input)
    #
    # # output = output[0][0].double()
    # res = torch.autograd.gradcheck (loss_fn, (input, output), eps=1e-6, raise_exception=True)
    print (output.shape)