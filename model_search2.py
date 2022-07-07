import torch
import torch.nn as nn
import torch.nn.functional as F
from operations2 import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype


class MixedOp(nn.Module):

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:  # 8
            op = OPS[primitive](C, C)
            self._ops.append(op)

    def forward(self, x, weights):
        # for op in self._ops:
            # print(op(x).shape)
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev, C, latency):
        super(Cell, self).__init__()
        self.preprocess = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier
        self._ops = nn.ModuleList()
        self.latency = latency
        for i in range(self._steps):  # 4个中间节点
            for j in range(1 + i):
                stride = 1
                op = MixedOp(C, stride)
                self._ops.append(op)  # 14个平均操作

    def forward(self, s, weights):
        s = self.preprocess(s)
        states = [s]
        offset = 0
        for i in range(self._steps):  # 对于每一个中间节点
            s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))  # 每个节点的多个平均操作求和，得到该点的输出
            offset += len(states)
            states.append(s)
        lat = 0
        for i in range(weights.shape[0]):
            lat += sum(self.latency[k] * weights[i][j] for j, k in enumerate(PRIMITIVES))
        return torch.cat(states[-self._multiplier:], dim=1), lat  # 合并4个节点的输出


class FM(nn.Module):

    def __init__(self, C, latency, steps=4, multiplier=4):
        super(FM, self).__init__()
        self._inC = C  # 4
        self._steps = steps
        self._multiplier = multiplier
        C_curr = C
        self.latency = latency
        self.cell_ir = Cell(steps, multiplier, 1, C_curr, latency['cell_en'])
        self.cell_vis = Cell(steps, multiplier, 1, C_curr, latency['cell_en'])
        self.cell_fu = Cell(steps, multiplier, C_curr*8, C_curr//2, latency['cell_de'])
        self.conv1x1 = nn.Conv2d(C_curr*2, 1, 1, padding=0, bias=True)
        self._initialize_alphas()

    def new(self):
        model_new = FM(self._inC).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input_ir, input_vis):
        weights_alpha1 = F.softmax(self.alphas1, dim=-1)
        feature_ir, lat1 = self.cell_ir(input_ir, weights_alpha1)
        weights_alpha2 = F.softmax(self.alphas2, dim=-1)
        feature_vis, lat2 = self.cell_vis(input_vis, weights_alpha2)
        weights_alpha3 = F.softmax(self.alphas3, dim=-1)
        feature_fusion, lat3 = self.cell_fu(torch.cat([feature_ir, feature_vis], dim=1), weights_alpha3)
        output = self.conv1x1(feature_fusion)
        return output, (lat1 + lat2 + lat3)

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(1 + i))
        num_ops = len(PRIMITIVES)
        self.alphas1 = Variable(1e-3 * torch.randn((k, num_ops))).cuda()
        self.alphas1.requires_grad = True
        self.alphas2 = Variable(1e-3 * torch.randn((k, num_ops))).cuda()
        self.alphas2.requires_grad = True
        self.alphas3 = Variable(1e-3 * torch.randn((k, num_ops))).cuda()
        self.alphas3.requires_grad = True
        self._arch_parameters = [
            self.alphas1,
            self.alphas2,
            self.alphas3
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):
        def _parse(weights):
            gene = []
            n = 1
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 1),
                               key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[
                        :2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        gene1 = _parse(F.softmax(self.alphas1, dim=-1).data.cpu().numpy())
        concat = range(1 + self._steps - self._multiplier, self._steps + 1)
        genotype1 = Genotype(
            cell=gene1, cell_concat=concat
        )
        gene2 = _parse(F.softmax(self.alphas2, dim=-1).data.cpu().numpy())
        genotype2 = Genotype(
            cell=gene2, cell_concat=concat
        )
        gene3 = _parse(F.softmax(self.alphas3, dim=-1).data.cpu().numpy())
        genotype3 = Genotype(
            cell=gene3, cell_concat=concat
        )
        return genotype1, genotype2, genotype3



