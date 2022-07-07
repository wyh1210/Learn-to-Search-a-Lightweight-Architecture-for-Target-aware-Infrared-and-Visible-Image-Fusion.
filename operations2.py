import torch
import torch.nn as nn

OPS = {
    'skip_connect': lambda C_in, C_out: Identity(),
    'none': lambda C_in, C_out: Zero(),
    'conv_1x1': lambda C_in, C_out: ConvBlock(C_in, C_out, 1),
    'conv_3x3': lambda C_in, C_out: ConvBlock(C_in, C_out, 3),
    'conv_5x5': lambda C_in, C_out: ConvBlock(C_in, C_out, 5),
    'conv_7x7': lambda C_in, C_out: ConvBlock(C_in, C_out, 7),
    'dilconv_3x3': lambda C_in, C_out: ConvBlock(C_in, C_out, 3, dilation=2),
    'dilconv_5x5': lambda C_in, C_out: ConvBlock(C_in, C_out, 5, dilation=2),
    'dilconv_7x7': lambda C_in, C_out: ConvBlock(C_in, C_out, 7, dilation=2),
    'resconv_1x1': lambda C_in, C_out: ResBlock(C_in, C_out, 1),
    'resconv_3x3': lambda C_in, C_out: ResBlock(C_in, C_out, 3),
    'resconv_5x5': lambda C_in, C_out: ResBlock(C_in, C_out, 5),
    'resconv_7x7': lambda C_in, C_out: ResBlock(C_in, C_out, 7),
    'resdilconv_3x3': lambda C_in, C_out: ResBlock(C_in, C_out, 3, dilation=2),
    'resdilconv_5x5': lambda C_in, C_out: ResBlock(C_in, C_out, 5, dilation=2),
    'resdilconv_7x7': lambda C_in, C_out: ResBlock(C_in, C_out, 7, dilation=2),
}


class ConvBlock(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride=1, dilation=1, groups=1):
        super(ConvBlock, self).__init__()
        padding = int((kernel_size - 1) / 2) * dilation
        self.op = nn.Conv2d(C_in, C_out, kernel_size, stride, padding=padding, bias=True, dilation=dilation, groups=groups, padding_mode='reflect')
        self.bn = nn.BatchNorm2d(C_out, eps=1e-5, momentum=0.01, affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        return self.relu(self.bn(self.op(x)))


class ResBlock(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride=1, dilation=1, groups=1):
        super(ResBlock, self).__init__()
        padding = int((kernel_size - 1) / 2) * dilation
        self.conv1x1 = nn.Conv2d(C_in, C_out, kernel_size=1)
        self.op = nn.Conv2d(C_out, C_out, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                            groups=groups, padding_mode='reflect')
        self.bn = nn.BatchNorm2d(C_out, eps=1e-5, momentum=0.01, affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv1x1(x)
        return self.relu(self.bn(self.op(x))) + x


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False, padding_mode='reflect'),
            # nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class Zero(nn.Module):
    def __init__(self, stride=1):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)
