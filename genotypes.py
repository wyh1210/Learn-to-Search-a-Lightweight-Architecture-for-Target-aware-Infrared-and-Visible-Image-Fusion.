from collections import namedtuple

Genotype = namedtuple('Genotype', 'cell cell_concat')

PRIMITIVES = [
    # 'skip_connect': lambda C_in, C_out: Identity(),
    'none',
    'conv_1x1',
    'conv_3x3',
    'conv_5x5',
    # 'conv_7x7',
    'dilconv_3x3',
    'dilconv_5x5',
    # 'dilconv_7x7',
    'resconv_1x1',
    'resconv_3x3',
    'resconv_5x5',
    # 'resconv_7x7',
    'resdilconv_3x3',
    'resdilconv_5x5',
    # 'resdilconv_7x7'
]



# genotype1 = Genotype(cell=[('conv_1x1', 0), ('resconv_3x3', 1), ('conv_1x1', 0), ('resconv_3x3', 2), ('resconv_3x3', 1), ('resdilconv_3x3', 3), ('conv_3x3', 1)], cell_concat=range(1, 5))
# genotype2 = Genotype(cell=[('resdilconv_5x5', 0), ('dilconv_3x3', 1), ('resdilconv_3x3', 0), ('resconv_1x1', 2), ('resdilconv_3x3', 0), ('resconv_3x3', 3), ('conv_1x1', 0)], cell_concat=range(1, 5))
# genotype3 = Genotype(cell=[('resconv_5x5', 0), ('resdilconv_3x3', 0), ('resdilconv_5x5', 1), ('resdilconv_5x5', 0), ('resdilconv_3x3', 1), ('resconv_3x3', 3), ('resconv_5x5', 0)], cell_concat=range(1, 5))



# lambda = 0.01
# genotype1 = Genotype(cell=[('resconv_1x1', 0), ('resconv_5x5', 0), ('conv_1x1', 1), ('conv_5x5', 2), ('conv_1x1', 0), ('conv_1x1', 1), ('conv_1x1', 0)], cell_concat=range(1, 5))
# genotype2 = Genotype(cell=[('conv_1x1', 0), ('conv_1x1', 0), ('resconv_3x3', 1), ('resdilconv_3x3', 2), ('resdilconv_3x3', 1), ('resconv_3x3', 0), ('resconv_5x5', 3)], cell_concat=range(1, 5))
# genotype3 = Genotype(cell=[('resdilconv_3x3', 0), ('resconv_1x1', 1), ('resdilconv_3x3', 0), ('resconv_1x1', 0), ('resconv_3x3', 2), ('resconv_1x1', 0), ('resdilconv_5x5', 1)], cell_concat=range(1, 5))

# # lambda = 0.1
genotype1 = Genotype(cell=[('conv_1x1', 0), ('conv_3x3', 1), ('conv_1x1', 0), ('resdilconv_5x5', 1), ('resconv_3x3', 2), ('conv_1x1', 1), ('conv_1x1', 0)], cell_concat=range(1, 5))
genotype2 = Genotype(cell=[('conv_5x5', 0), ('resconv_1x1', 0), ('resconv_3x3', 1), ('conv_1x1', 2), ('resconv_1x1', 0), ('resconv_1x1', 1), ('dilconv_5x5', 0)], cell_concat=range(1, 5))
genotype3 = Genotype(cell=[('conv_3x3', 0), ('conv_1x1', 0), ('resconv_1x1', 1), ('resconv_5x5', 2), ('resconv_1x1', 0), ('conv_3x3', 2), ('resdilconv_3x3', 0)], cell_concat=range(1, 5))


# lambda = 1
# genotype1 = Genotype(cell=[('resconv_1x1', 0), ('conv_1x1', 1), ('conv_1x1', 0), ('conv_1x1', 2), ('conv_1x1', 1), ('conv_1x1', 0), ('conv_1x1', 1)], cell_concat=range(1, 5))
# genotype2 = Genotype(cell=[('resconv_1x1', 0), ('conv_1x1', 0), ('resconv_3x3', 1), ('conv_5x5', 0), ('conv_1x1', 1), ('resconv_3x3', 0), ('resdilconv_3x3', 2)], cell_concat=range(1, 5))
# genotype3 = Genotype(cell=[('resdilconv_3x3', 0), ('conv_1x1', 0), ('resconv_1x1', 1), ('conv_5x5', 0), ('resconv_3x3', 2), ('conv_1x1', 1), ('resconv_1x1', 0)], cell_concat=range(1, 5))

# lambda = 0.5
# genotype1 = Genotype(cell=[('conv_3x3', 0), ('conv_1x1', 0), ('resconv_3x3', 1), ('resconv_3x3', 0), ('dilconv_5x5', 1), ('dilconv_3x3', 3), ('conv_1x1', 0)], cell_concat=range(1, 5))
# genotype2 = Genotype(cell=[('resconv_1x1', 0), ('resconv_1x1', 0), ('resconv_3x3', 1), ('resdilconv_3x3', 0), ('conv_1x1', 1), ('resconv_3x3', 0), ('resdilconv_3x3', 2)], cell_concat=range(1, 5))
# genotype3 = Genotype(cell=[('dilconv_5x5', 0), ('resdilconv_3x3', 0), ('resconv_1x1', 1), ('resconv_5x5', 2), ('resconv_1x1', 0), ('resconv_3x3', 2), ('resdilconv_3x3', 0)], cell_concat=range(1, 5))

# lambda = 0.05
# genotype1 = Genotype(cell=[('conv_3x3', 0), ('conv_1x1', 0), ('resconv_3x3', 1), ('conv_3x3', 0), ('dilconv_3x3', 1), ('resconv_1x1', 0), ('conv_1x1', 1)], cell_concat=range(1, 5))
# genotype2 = Genotype(cell=[('conv_1x1', 0), ('dilconv_3x3', 0), ('resdilconv_5x5', 1), ('resconv_1x1', 1), ('conv_5x5', 0), ('dilconv_5x5', 0), ('resconv_5x5', 3)], cell_concat=range(1, 5))
# genotype3 = Genotype(cell=[('resdilconv_3x3', 0), ('resconv_1x1', 1), ('conv_1x1', 0), ('resconv_1x1', 0), ('resconv_3x3', 1), ('resconv_1x1', 0), ('conv_1x1', 1)], cell_concat=range(1, 5))






