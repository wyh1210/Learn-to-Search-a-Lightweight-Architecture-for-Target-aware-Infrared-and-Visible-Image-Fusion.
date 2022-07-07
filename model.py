import torch
import torch.nn as nn
from operations2 import *
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


class Cell(nn.Module):

    def __init__(self, genotype, C_prev, C):
        super(Cell, self).__init__()
        print(C_prev, C)

        self.preprocess = ReLUConvBN(C_prev, C, 1, 1, 0)
        op_names, indices = zip(*genotype.cell)
        concat = genotype.cell_concat
        self._compile(C, op_names, indices, concat)

    def _compile(self, C, op_names, indices, concat):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)
        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            op = OPS[name](C, C)
            self._ops += [op]
        self._indices = indices

    def forward(self, s):
        s = self.preprocess(s)
        states = [s]
        h = states[0]
        op = self._ops[0]
        s = op(h)
        states += [s]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i + 1]
            op2 = self._ops[2 * i + 2]
            h1 = op1(h1)
            h2 = op2(h2)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)


class FM(nn.Module):

    def __init__(self, C, genotype1, genotype2, genotype3):
        super(FM, self).__init__()
        self._inC = C  # 4
        self.cell_ir = Cell(genotype1, 1, C)
        self.cell_vis = Cell(genotype2, 1, C)
        self.cell_fu = Cell(genotype3, C*8, C//2)
        self.conv1x1 = nn.Conv2d(C*2, 1, 1, padding=0, bias=True)

    def forward(self, input_ir, input_vis):
        feature_ir = self.cell_ir(input_ir)
        feature_vis = self.cell_vis(input_vis)
        feature_fusion = self.cell_fu(torch.cat([feature_ir, feature_vis], dim=1))
        output = self.conv1x1(feature_fusion)
        return output


class Commonlayer(nn.Module):

    def __init__(self, inc, outc, kernel_size, stride):
        super(Commonlayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(inc, outc, kernel_size, stride)
        self.bn = nn.BatchNorm2d(outc, affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.reflection_pad(x)
        x = self.conv2d(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Block(nn.Module):

    def __init__(self, inc, outc, kernel_size, stride):
        super(Block, self).__init__()
        self.layer1 = Commonlayer(inc, outc, kernel_size, stride)
        self.layer2 = Commonlayer(outc, outc, kernel_size, stride)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = torch.cat([x, out], 1)
        return out


class Commonlayer2(nn.Module):

    def __init__(self, inc, outc, kernel_size, stride):
        super(Commonlayer2, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(inc, outc, kernel_size, stride)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.reflection_pad(x)
        x = self.conv2d(x)
        x = self.relu(x)
        return x


class Commonlayer5(nn.Module):

    def __init__(self, inc, outc, kernel_size, stride):
        super(Commonlayer5, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(inc, outc, kernel_size, stride)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.reflection_pad(x)
        x = self.conv2d(x)
        x = self.tanh(x)
        return x


class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        self.common_layer1 = Commonlayer(2, 48, 3, 1)
        self.block1 = Block(48, 48, 3, 1)
        self.block2 = Block(96, 48, 3, 1)
        self.block3 = Block(144, 48, 3, 1)
        self.block4 = Block(192, 48, 3, 1)
        self.common_layer2 = Commonlayer2(240, 240, 3, 1)
        self.common_layer3 = Commonlayer(240, 128, 3, 1)
        self.common_layer4 = Commonlayer(128, 64, 3, 1)
        self.common_layer5 = Commonlayer5(64, 1, 3, 1)

    def forward(self, x):
        out = self.common_layer1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.common_layer2(out)
        out = self.common_layer3(out)
        out = self.common_layer4(out)
        out = self.common_layer5(out)
        return out



