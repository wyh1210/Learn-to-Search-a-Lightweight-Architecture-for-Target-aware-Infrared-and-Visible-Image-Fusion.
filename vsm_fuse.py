from os.path import join
from os import listdir
import PIL.Image as Image
import numpy as np
import os
from os.path import exists
from utils import get_train_images_auto, get_test_images
import cv2
import genotypes


def calhis(img):
    ret = np.zeros(256, int)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            ret[img[i][j]] += 1
    return ret
def sal(his):
    ret = np.zeros(256, int)
    for i in range(256):
        for j in range(256):
            ret[i] += np.abs(j-i)*his[j]
    return ret

def vsm(img):
    his = np.zeros(256, int)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            his[img[i][j]] += 1
    sal = np.zeros(256, int)
    for i in range(256):
        for j in range(256):
            sal[i] += np.abs(j-i)*his[j]
    map = np.zeros_like(img, int)
    for i in range(256):
        map[np.where(img == i)] = sal[i]
    return map / (map.max())


image_dir1 = r'C:\Users\ADMIN\Desktop\test_data\TNO'
image_dir2 = r'C:\Users\ADMIN\Desktop\test_data\Filr'


save_dir = r'C:\Users\ADMIN\Desktop\eff_net\softmax1_fuse_result'


if not exists(save_dir):
    os.mkdir(save_dir)


def softmax(map1, map2, c):
    exp_x1 = np.exp(map1*c)
    exp_x2 = np.exp(map2*c)
    exp_sum = exp_x1 + exp_x2
    map1 = exp_x1/exp_sum
    map2 = exp_x2/exp_sum
    print(map1)
    print(map2)
    return  map1, map2



for i in range(37):
    image_ir_path = join(image_dir1, 'ir', str(i+1)+'.bmp')
    image_vis_path = join(image_dir1, 'vis', str(i+1)+'.bmp')

    img_ir = np.asarray(Image.open(image_ir_path).convert('L'))
    img_vis = np.asarray(Image.open(image_vis_path).convert('L'))
    map1 = vsm(img_ir)
    map2 = vsm(img_vis)
    # w1 = 0.5 + 0.5 * (map1 - map2)  # 红外光图像的显著性map
    # w2 = 0.5 + 0.5 * (map2 - map1)  # 可见光
    # print(w1.shape)
    # print(img_ir.shape)
    w1, w2 = softmax(map1, map2, c=1)

    img_fuse = w1*img_ir + w2*img_vis
    cv2.imwrite(join(save_dir, 'TNO_'+str(i+1)+'.jpg'), img_fuse)
    # img_fuse.save(join(save_dir, 'TNO_'+str(i+1)+'.jpg'))

# for i in range(42):
#     image_ir_path = join(image_dir2, 'Ir' + str(i+1)+'.jpg')
#     image_vis_path = join(image_dir2, 'Vis' + str(i+1)+'.jpg')
#
#     # print(image_ir)
#     # print(image_vis)
#
#     tensor_ir = get_test_images(image_ir_path).cuda()
#     # print(tensor_ir)
#     tensor_vis = get_test_images(image_vis_path).cuda()
#
#     tensor_f = model(tensor_ir, tensor_vis).cpu()
#
#     image_tensor = tensor_f.squeeze()  # .squeeze()
#     image_tensor = torch.clamp(image_tensor, 0, 1)
#     # print(image_tensor.size())
#     # print(image_tensor)
#     print(i)
#     image_pil = tensor_to_pil(image_tensor)
#     image_pil.save(join(save_dir, 'Filr_'+str(i+1)+'.jpg'))


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

