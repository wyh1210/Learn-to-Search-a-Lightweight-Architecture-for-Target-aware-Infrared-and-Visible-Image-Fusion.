import pickle
from genotypes import PRIMITIVES
with open(r'./latency_gpu_de.pkl', 'rb') as f:
    lat = pickle.load(f)
    print(len(lat), len(lat[0]))
    print(sum(lat[0][k] for j, k in enumerate(PRIMITIVES)))
    for i in PRIMITIVES:
        print(lat[0][i])
    # print(lat[4]['resdilconv_5x5'])

# for j, k in enumerate(PRIMITIVES):
#     print(j, k)