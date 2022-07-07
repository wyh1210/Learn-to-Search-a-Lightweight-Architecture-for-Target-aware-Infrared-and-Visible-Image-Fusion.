import torch
import torch.nn as nn
import torch.nn.functional as F
from operations2 import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype


class MixedOp(nn.Module):

    def __init__(self, C_in, C_out):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:  # 8
            op = OPS[primitive](C_in, C_out)
            self._ops.append(op)

    def forward(self, x, weights):
        # for op in self._ops:
            # print(op(x).shape)
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class BlockEn(nn.Module):

    def __init__(self, latency, steps=9, C=[4, 8, 16, 16, 32, 32, 48, 48, 64]):
        super(BlockEn, self).__init__()
        self.latency = latency
        self._steps = steps  # 9
        self._ops = nn.ModuleList()
        C_in = 1
        for i in range(self._steps):  # 4个中间节点
            C_out = C[i]
            op = MixedOp(C_in, C_out)
            self._ops.append(op)  # 14个平均操作
            C_in = C_out

    def forward(self, x, weights):
        lat = 0
        for i in range(self._steps):  # 对于每一个中间节点
            # print('i', i)
            x = self._ops[i](x, weights[i])
            lat += sum(self.latency[i][k]*weights[i][j] for j, k in enumerate(PRIMITIVES))
        return x, lat


class BlockDe(nn.Module):
    def __init__(self, latency, steps=9, C=[64, 48, 48, 32, 32, 16, 16, 8, 1]):
        super(BlockDe, self).__init__()
        self.latency = latency
        self._steps = steps  # 9
        self._ops = nn.ModuleList()
        C_in = 128
        for i in range(self._steps):  # 4个中间节点
            C_out = C[i]
            op = MixedOp(C_in, C_out)
            self._ops.append(op)  # 14个平均操作
            C_in = C_out

    def forward(self, x, weights):
        lat = 0
        for i in range(self._steps):  # 对于每一个中间节点
            x = self._ops[i](x, weights[i])
            lat += sum(self.latency[i][k]*weights[i][j] for j, k in enumerate(PRIMITIVES))
        return x, lat


class Network(nn.Module):
    def __init__(self, latencyEn, latencyDe):
        super(Network, self).__init__()
        self._steps = 9
        self.en1 = BlockEn(latencyEn)
        self.en2 = BlockEn(latencyEn)
        self.de = BlockDe(latencyDe)
        self._initialize_alphas()

    def forward(self, img1, img2):
        weights1 = F.softmax(self.alphas1, dim=-1)
        fe1, l1 = self.en1(img1, weights1)
        weights2 = F.softmax(self.alphas2, dim=-1)
        fe2, l2 = self.en2(img2, weights2)
        weights3 = F.softmax(self.alphas3, dim=-1)
        fe_concat = torch.cat([fe1, fe2], dim=1)
        out, l3 = self.de(fe_concat, weights3)
        lat = l1 + l2 + l3
        return out, lat

    def new(self):
        model_new = Network().cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def _initialize_alphas(self):
        k = 9
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
            print(weights.shape)
            gene = []
            for i in range(self._steps):
                ind = sorted(range(weights.shape[1]), key=lambda x: -weights[i][x])[0]
                gene.append(PRIMITIVES[ind])
            return gene
        gene_en1 = _parse(F.softmax(self.alphas1, dim=-1).data.cpu().numpy())
        concat = range(9)
        genotype1 = Genotype(
            cell=gene_en1, cell_concat=concat
        )
        gene_en1 = _parse(F.softmax(self.alphas2, dim=-1).data.cpu().numpy())
        genotype2 = Genotype(
            cell=gene_en1, cell_concat=concat
        )
        gene_en1 = _parse(F.softmax(self.alphas3, dim=-1).data.cpu().numpy())
        genotype3 = Genotype(
            cell=gene_en1, cell_concat=concat
        )
        return genotype1, genotype2, genotype3


# class Encoder(nn.Module):
#
#     def __init__(self, C, layers, steps=4, multiplier=4):
#         super(Encoder, self).__init__()
#         self._inC = C  # 4
#         self._layers = layers  # 3
#         self._steps = steps
#         self._multiplier = multiplier
#         C_curr = 8
#
#         self.stem = nn.Sequential(
#             nn.ReflectionPad2d(1),
#             nn.Conv2d(1, 8, 3, padding=0, bias=False),
#             # nn.BatchNorm2d(8)
#         )
#
#         C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
#         self.cells = nn.ModuleList()
#         for i in range(layers):
#             # C_curr = C*(2**i)
#             cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr)
#             self.cells += [cell]
#             C_prev_prev, C_prev = C_prev, multiplier * C_curr
#
#         self._initialize_alphas()
#
#     def new(self):
#         model_new = Encoder(self._inC, self._layers).cuda()
#         for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
#             x.data.copy_(y.data)
#         return model_new
#
#     def forward(self, input):
#         s0 = s1 = self.stem(input)
#         for i, cell in enumerate(self.cells):
#             weights = F.softmax(self.alphas, dim=-1)
#             s0, s1 = s1, cell(s0, s1, weights)
#         return s0, s1
#
#     def _initialize_alphas(self):
#         k = sum(1 for i in range(self._steps) for n in range(2 + i))  # 14
#         num_ops = len(PRIMITIVES)
#
#         self.alphas = Variable(1e-3 * torch.randn((k, num_ops))).cuda()
#         self.alphas.requires_grad = True
#
#         self._arch_parameters = [
#             self.alphas
#         ]
#
#     def arch_parameters(self):
#         return self._arch_parameters
#
#     def genotype(self):
#         def _parse(weights):
#             gene = []
#             n = 2
#             start = 0
#             for i in range(self._steps):
#                 end = start + n
#                 W = weights[start:end].copy()
#                 edges = sorted(range(i + 2),
#                                key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[
#                         :2]
#                 for j in edges:
#                     k_best = None
#                     for k in range(len(W[j])):
#                         if k != PRIMITIVES.index('none'):
#                             if k_best is None or W[j][k] > W[j][k_best]:
#                                 k_best = k
#                     gene.append((PRIMITIVES[k_best], j))
#                 start = end
#                 n += 1
#             return gene
#
#         gene_former = _parse(F.softmax(self.alphas, dim=-1).data.cpu().numpy())
#         concat = range(2 + self._steps - self._multiplier, self._steps + 2)
#         genotype = Genotype(
#             cell=gene_former, cell_concat=concat
#         )
#         return genotype
#
#
# class Decoder(nn.Module):
#
#     def __init__(self, C, layers, steps=4, multiplier=4):
#         super(Decoder, self).__init__()
#         self._inC = C  # 8
#         self._layers = layers  # 2
#         self._steps = steps
#         self._multiplier = multiplier
#
#         C_prev_prev, C_prev, C_curr = C*4, C*4, C
#         self.cells = nn.ModuleList()
#         for i in range(layers):
#             # C_curr = C//(2**i)
#             cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr)
#             self.cells += [cell]
#             C_prev_prev, C_prev = C_prev, multiplier * C_curr
#         self.pad = nn.ReflectionPad2d(1)
#         self.ConvLayer = nn.Conv2d(C_curr*multiplier, 1, 3, padding=0)
#         # self.tanh = nn.Tanh()
#         self._initialize_alphas()
#
#     def new(self):
#         model_new = Decoder(self._inC, self._layers).cuda()
#         for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
#             x.data.copy_(y.data)
#         return model_new
#
#     def forward(self, s0, s1):
#         for i, cell in enumerate(self.cells):
#             weights = F.softmax(self.alphas, dim=-1)
#             s0, s1 = s1, cell(s0, s1, weights)
#         output = self.pad(s1)
#         output = self.ConvLayer(output)
#         return output
#
#     def _initialize_alphas(self):
#         k = sum(1 for i in range(self._steps) for n in range(2 + i))  # 14
#         num_ops = len(PRIMITIVES)
#
#         self.alphas = Variable(1e-3 * torch.randn((k, num_ops))).cuda()
#         self.alphas.requires_grad = True
#
#         self._arch_parameters = [
#             self.alphas
#         ]
#
#     def arch_parameters(self):
#         return self._arch_parameters
#
#     def genotype(self):
#         def _parse(weights):
#             gene = []
#             n = 2
#             start = 0
#             for i in range(self._steps):
#                 end = start + n
#                 W = weights[start:end].copy()
#                 edges = sorted(range(i + 2),
#                                key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[
#                         :2]
#                 for j in edges:
#                     k_best = None
#                     for k in range(len(W[j])):
#                         if k != PRIMITIVES.index('none'):
#                             if k_best is None or W[j][k] > W[j][k_best]:
#                                 k_best = k
#                     gene.append((PRIMITIVES[k_best], j))
#                 start = end
#                 n += 1
#             return gene
#
#         gene = _parse(F.softmax(self.alphas, dim=-1).data.cpu().numpy())
#         concat = range(2 + self._steps - self._multiplier, self._steps + 2)
#         genotype = Genotype(
#             cell=gene, cell_concat=concat
#         )
#         return genotype


