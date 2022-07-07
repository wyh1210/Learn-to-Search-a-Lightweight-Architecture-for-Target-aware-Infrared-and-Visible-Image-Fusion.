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


model_dir = r'C:\Users\ADMIN\Desktop\eff_net\trainTNORoadSceneSoftmax_0.1-20210929-115733'
model_path = join(model_dir, 'weights_epoch_50.pt')

genotype_en1 = eval('genotypes.%s' % 'genotype1')
genotype_en2 = eval('genotypes.%s' % 'genotype2')
genotype_de = eval('genotypes.%s' % 'genotype3')

model = FM(16, genotype_en1, genotype_en2, genotype_de).cuda()


params = torch.load(model_path)

model.load_state_dict(params)

model.eval()


image_dir1 = r'C:\Users\ADMIN\Desktop\test_data\TNO'
image_dir2 = r'C:\Users\ADMIN\Desktop\test_data\Filr'


save_dir = r'C:\Users\ADMIN\Desktop\eff_net\0.01(2)test_softmaxep50'


if not exists(save_dir):
    os.mkdir(save_dir)


with torch.no_grad():
    total1 = 0
    for i in range(37):
        image_ir_path = join(image_dir1, 'ir', str(i+1)+'.bmp')
        image_vis_path = join(image_dir1, 'vis', str(i+1)+'.bmp')

        # print(image_ir)
        # print(image_vis)

        tensor_ir = get_test_images(image_ir_path).cuda()
        # print(tensor_ir)
        tensor_vis = get_test_images(image_vis_path).cuda()

        t1 = time.time()
        tensor_f = model(tensor_ir, tensor_vis)
        t2 = time.time()

        image_tensor = tensor_f.cpu().squeeze()  # .squeeze()
        image_tensor = torch.clamp(image_tensor, 0, 1)
        # print(image_tensor.size())
        # print(image_tensor)
        print(i)
        image_pil = tensor_to_pil(image_tensor)
        # image_pil.save(join(save_dir, 'TNO_'+str(i+1)+'.jpg'))
        total1 += t2 - t1
    total2 = 0
    for i in range(42):
        image_ir_path = join(image_dir2, 'Ir' + str(i+1)+'.jpg')
        image_vis_path = join(image_dir2, 'Vis' + str(i+1)+'.jpg')

        # print(image_ir)
        # print(image_vis)

        tensor_ir = get_test_images(image_ir_path).cuda()
        # print(tensor_ir)
        tensor_vis = get_test_images(image_vis_path).cuda()

        t1 = time.time()
        tensor_f = model(tensor_ir, tensor_vis)
        t2 = time.time()

        image_tensor = tensor_f.cpu().squeeze()  # .squeeze()
        image_tensor = torch.clamp(image_tensor, 0, 1)
        # print(image_tensor.size())
        # print(image_tensor)
        print(i)
        image_pil = tensor_to_pil(image_tensor)
        # image_pil.save(join(save_dir, 'Filr_'+str(i+1)+'.jpg'))
        total2 += t2 - t1


    print("param size = %fMB", utils.count_parameters_in_MB(model))
    print('TNO consuming time: {} Filr consuming time: {}'.format(total1, total2))
    print('Time cost per image pair:|| TNO: {}, Filr: {}'.format(total1/37, total2/42))
    # 上色
    # namelist = os.listdir(image_dir2+'\\ir')
    #
    # # print(namelist)
    # for name in namelist:
    #     ir = cv2.imread(os.path.join(image_dir2, 'ir', name))
    #     # print(ir.shape)
    #     vis = cv2.imread(os.path.join(image_dir2, 'vis_rgb', name))
    #     # print(ir.shape)
    #     # print(vis.shape)
    #     # print(ir[:, :, 0])
    #     # print(ir[:, :, 1])
    #     # print(ir[:, :, 2])
    #     vis_ycrcb = cv2.cvtColor(vis, cv2.COLOR_BGR2YCrCb)
    #
    #     tensor1 = torch.tensor(ir[:, :, 0], dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
    #     tensor2 = torch.tensor(vis_ycrcb[:, :, 0], dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
    #     # print(tensor1.shape)
    #     # print(tensor2.shape)
    #
    #     input = torch.cat([tensor1, tensor2], 1)
    #
    #     # print('input:',input.shape)
    #     tensor_f = model(input).cpu()
    #
    #     image_tensor = tensor_f.squeeze()  # .squeeze()
    #     image_tensor = torch.clamp(image_tensor, 0, 255)
    #     # print(image_tensor.size())
    #     # print(i)
    #     image_array = np.asarray(image_tensor.detach(), dtype=int)
    #     # print(image_array.shape)
    #     # print(vis_ycrcb.shape)
    #     re = np.stack([image_array, vis_ycrcb[:, :, 1], vis_ycrcb[:, :, 2]], axis=2).astype(np.uint8)
    #     # print(re.shape, re.dtype)
    #     re = cv2.cvtColor(re, cv2.COLOR_YCrCb2BGR)
    #     cv2.imwrite(save_dir + '\\Road_rgb'+name, re)
    #     cv2.imwrite(save_dir + '\\Road_' + name, image_array.astype(np.uint8))
