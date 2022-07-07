import torch
from model import FM
from os.path import join
from os import listdir
import PIL.Image as Image
import numpy as np
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import os
from os.path import exists
from utils import get_train_images_auto, get_test_images
import cv2
import genotypes
import utils
import time
tensor_to_pil = transforms.ToPILImage()


model_dir = r'E:\已投论文\ICASSP全\eff_net\trainTNORoadSceneSoftmax_0.1-20210929-115733'
model_path = join(model_dir, 'weights_epoch_100.pt')

genotype_en1 = eval('genotypes.%s' % 'genotype1')
genotype_en2 = eval('genotypes.%s' % 'genotype2')
genotype_de = eval('genotypes.%s' % 'genotype3')

model = FM(16, genotype_en1, genotype_en2, genotype_de).cuda()


params = torch.load(model_path)

model.load_state_dict(params)

model.eval()


image_dir1 = r'G:\DIVIF\Ir'
image_dir2 = r'G:\DIVIF\Vis'

save_dir = r'G:\DIVIF\WUICASSP'


if not exists(save_dir):
    os.mkdir(save_dir)


with torch.no_grad():
    namelist = os.listdir(image_dir1)
    for name in namelist:
        # print(name)

        ir = cv2.imread(os.path.join(image_dir1, name))
        # print(ir.shape)
        vis = cv2.imread(os.path.join(image_dir2, name))
        # print(vis)
        vis_ycrcb = cv2.cvtColor(vis, cv2.COLOR_BGR2YCrCb)

        tensor1 = torch.tensor(ir[:, :, 0], dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
        tensor2 = torch.tensor(vis_ycrcb[:, :, 0], dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()

        tensor_f = model(tensor1, tensor2).cpu()

        image_tensor = tensor_f.squeeze()  # .squeeze()
        image_tensor = torch.clamp(image_tensor, 0, 255)
        # print(image_tensor.size())
        # print(i)
        image_array = np.asarray(image_tensor.detach(), dtype=int)
        # print(image_array)
        re = np.stack([image_array, vis_ycrcb[:, :, 1], vis_ycrcb[:, :, 2]], axis=2).astype(np.uint8)
        # print(re.shape, re.dtype)
        re = cv2.cvtColor(re, cv2.COLOR_YCrCb2BGR)

        # name0, name1 = name.split('.')
        # print(name0, name1)

        cv2.imwrite(save_dir + '\\' + name, re)
        # cv2.imwrite(save_dir + '\\' + name0+'.jpg', image_array.astype(np.uint8))
