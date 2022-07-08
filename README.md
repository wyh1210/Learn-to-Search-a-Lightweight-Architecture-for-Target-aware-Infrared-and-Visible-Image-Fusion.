# Introduction
2
​
3
This is the implementation of the paper [Learn to Search a Lightweight Architecture for Target-aware Infrared and Visible Image Fusion](https://ieeexplore.ieee.org/document/9789723).
4
​
5
## Requirements
6
​
7
* python >= 3.6
8
* pytorch == 1.7
9
* torchvision == 0.8
10
​
11
## Datasets
12
​
13
You can download the datasets [here](https://pan.baidu.com/s/1Ckq5v-d2JpG8YsfqDUjz-A?pwd=siis).
14
​
15
## Test
16
​
17
```shell
18
python test.py
19
```
20
​
21
## Train from scratch
22
​
23
### step 1
24
​
25
```shell
26
python train_search.py
27
```
28
​
29
### step 2
30
​
31
Find the string which descripting the searched architectures in the log file. Copy and paste it into the genotypes.py, the format should consist with the primary architecture string.
32
​
33
### step 3
34
​
35
```shell
36
python train.py
37
```
38
​
39
## Citation
40
​
41
If you use any part of this code in your research, please cite our [paper](https://ieeexplore.ieee.org/document/9789723):
42
​
43
```
44
@ARTICLE{9789723,  author={Liu, Jinyuan and Wu, Yuhui and Wu, Guanyao and Liu, Risheng and Fan, Xin},  
journal={IEEE Signal Processing Letters},   
title={Learn to Search a Lightweight Architecture for Target-aware Infrared and Visible Image Fusion},   
year={2022},  
volume={},  
number={},  
pages={1-5},  
doi={10.1109/LSP.2022.3180672}}
45
```