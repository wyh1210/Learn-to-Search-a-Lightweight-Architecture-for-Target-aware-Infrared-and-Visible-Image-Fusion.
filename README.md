# Introduction

This is the implementation of the paper [Learn to Search a Lightweight Architecture for Target-aware Infrared and Visible Image Fusion](https://ieeexplore.ieee.org/document/9789723).

## Requirements

* python >= 3.6

* pytorch == 1.7

* torchvision == 0.8


## Datasets


You can download the datasets [here](https://pan.baidu.com/s/1Ckq5v-d2JpG8YsfqDUjz-A?pwd=siis).


## Test


```shell

python test.py

```

## Train from scratch

### step 1

```shell

python train_search.py

```

### step 2

Find the string which descripting the searched architectures in the log file. Copy and paste it into the genotypes.py, the format should consist with the primary architecture string.

### step 3

```shell

python train.py

```

## Citation


If you use any part of this code in your research, please cite our [paper](https://ieeexplore.ieee.org/document/9789723):

```

@ARTICLE{9789723,  
author={Liu, Jinyuan and Wu, Yuhui and Wu, Guanyao and Liu, Risheng and Fan, Xin},  
journal={IEEE Signal Processing Letters},   
title={Learn to Search a Lightweight Architecture for Target-aware Infrared and Visible Image Fusion},   
year={2022},  
volume={},  
number={},  
pages={1-5},  
doi={10.1109/LSP.2022.3180672}}

```
