import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from collections import OrderedDict
import pickle

sys.path.append('..')

from utils import measure_latency_in_ms
from operations2 import OPS
# from operations import OPS
cudnn.enabled = True
cudnn.benchmark = True

PRIMITIVES = [
    'none',
    'conv_1x1',
    'conv_3x3',
    'conv_5x5',
    'conv_7x7',
    'dilconv_3x3',
    'dilconv_5x5',
    'dilconv_7x7',
    'resconv_1x1',
    'resconv_3x3',
    'resconv_5x5',
    'resconv_7x7',
    'resdilconv_3x3',
    'resdilconv_5x5',
    'resdilconv_7x7',
]

# PRIMITIVES = {
#     'none',
#     'sep_conv_3x3',
#     'sep_conv_5x5',
#     # 'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
#     'dil_conv_3x3',
#     'dil_conv_5x5',
#     'NonLocalattention',
#     'Spatialattention',
#     'Denseblocks',
#     'Residualblocks',
# }


# def get_latency_lookup_en(is_cuda):
#     latency_lookup = OrderedDict()
#     C = [4, 8, 16, 16, 32, 32, 48, 48, 64]
#     C_in = 1
#     for i in range(9):
#         latency_lookup[i] = OrderedDict()
#         C_out = C[i]
#         for j in range(len(PRIMITIVES)):
#             op = OPS[PRIMITIVES[j]](C_in, C_out)
#             shape = [1, C_in, 128, 128]
#             print(shape)
#             lat = measure_latency_in_ms(op, shape, is_cuda)
#             latency_lookup[i][PRIMITIVES[j]] = lat
#         C_in = C_out
#     print(latency_lookup)
#     return latency_lookup
#
#
# def get_latency_lookup_de(is_cuda):
#     latency_lookup = OrderedDict()
#     C = [64, 48, 48, 32, 32, 16, 16, 8, 1]
#     C_in = 128
#     for i in range(9):
#         latency_lookup[i] = OrderedDict()
#         C_out = C[i]
#         for j in range(len(PRIMITIVES)):
#             op = OPS[PRIMITIVES[j]](C_in, C_out)
#             shape = [1, C_in, 128, 128]
#             print(shape)
#             lat = measure_latency_in_ms(op, shape, is_cuda)
#             latency_lookup[i][PRIMITIVES[j]] = lat
#         C_in = C_out
#     print(latency_lookup)
#     return latency_lookup


def get_latency_lookup_en(is_cuda):
    latency_lookup = OrderedDict()

    for type, C in zip(['cell_en', 'cell_de'], [16, 8]):
        latency_lookup[type] = OrderedDict()
        for j in range(len(PRIMITIVES)):
            op = OPS[PRIMITIVES[j]](C, C)
            shape = [1, C, 128, 128]
            print(shape)
            lat = measure_latency_in_ms(op, shape, is_cuda)
            latency_lookup[type][PRIMITIVES[j]] = lat

    print(latency_lookup)
    return latency_lookup


if __name__ == '__main__':
    print('measure latency on gpu......')
    latency_lookup = get_latency_lookup_en(is_cuda=True)
    # latency_lookup = convert_latency_lookup(latency_lookup)
    with open('latency_gpu.pkl', 'wb') as f:
        pickle.dump(latency_lookup, f)
        # lat = pickle.load(f)
    # print(lat)

    # print('measure latency on cpu......')
    # latency_lookup = get_latency_lookup(is_cuda=False)
    # # latency_lookup = convert_latency_lookup(latency_lookup)
    # with open('latency_cpu_example.pkl', 'wb') as f:
    #     pickle.dump(latency_lookup, f)
