import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from model import DenseNet
import torch.nn.functional as F
from os.path import join
from os import listdir
import random
from PIL import Image
import pytorch_msssim
import math
import torchvision.transforms as transform
parser = argparse.ArgumentParser("untitled")
import genotypes

parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='init learning rate')  # 0.025
# parser.add_argument('--learning_rate_min', type=float, default=0, help='min learning rate')  #0.001-->1e-4
# parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
# parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
# parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=60, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=8, help='num of init channels')

parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')

parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')

parser.add_argument('--dataset1', type=str, default=r'C:\Users\ADMIN\Desktop\roadSceneTNO128\crop_infrared128_20', help='Infrared images for training')
parser.add_argument('--dataset2', type=str, default=r'C:\Users\ADMIN\Desktop\roadSceneTNO128\crop_visible128_20', help='Visible images for training')

args = parser.parse_args()
args.save = 'trainDense_vsm-{}'.format(time.strftime("%Y%m%d-%H%M%S"))

utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

os.mkdir(args.save+'/output')


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True  # 加速
    torch.manual_seed(args.seed)  # 为CUP设置随机种子
    cudnn.enabled = True  # 使用非确定性算法优化运行

    torch.cuda.manual_seed(args.seed)  # 为GPU设置随机种子
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)
    mse_loss = torch.nn.MSELoss().cuda()
    ssim = pytorch_msssim.msssim

    model = DenseNet().cuda()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.6)

    epochs = args.epochs

    # 加载数据集
    Infrared_path_list = utils.list_images(args.dataset1)
    Visible_path_list = utils.list_images(args.dataset2)
    dir = r'C:\Users\ADMIN\Desktop\roadSceneTNO128\vsm'

    vsm_list = [os.path.join(dir, name) for name in listdir(dir)]
    imgQueue = np.stack([Infrared_path_list, Visible_path_list, vsm_list], axis=1)
    # print(vsm_list)
    # print(imgQueue)
    random.shuffle(imgQueue)
    train_queue, batches = utils.load_dataset(imgQueue, args.batch_size)

    for epoch in range(epochs):

        # training
        train(train_queue, batches, args, model, ssim, mse_loss, optimizer, epoch)

        if (epoch+1) % 5 == 0:
            utils.save(model, os.path.join(args.save, 'weights_epoch_' + str(epoch+1) + '.pt'))

        # scheduler.step()


tensor_to_pil = transform.ToPILImage()
pil_to_tensor = transform.ToTensor()


def train(train_queue, batches, args, model, ssim, mse_loss, optimizer, epoch):
    for batch in range(batches):
        paths_train = train_queue[batch * args.batch_size:(batch * args.batch_size + args.batch_size)]  # 训练一批
        train_batch = utils.get_batch(paths_train)
        # print(train_batch[0].shape, train_batch[1].shape, train_batch[2][0])

        tensor_ir, tensor_vis, map_list = train_batch[0].cuda(), train_batch[1].cuda(), train_batch[2]
        input_tensor = torch.cat([tensor_ir, tensor_vis], dim=1)
        outputs = model(input_tensor)

        if epoch > -1 and (batch + 1) % 10 == 0:
            output_tmp = outputs.detach().cpu()
            # tensor = torch.squeeze(output_tmp[0])
            tensor = torch.squeeze(output_tmp[0][0])
            # array = np.asarray(tensor)
            # image = Image.fromarray(array)
            # image = image.convert('L')
            # print(tensor)
            image = tensor_to_pil(tensor)
            image.save(args.save + '/output/image_batch' + str(batch + 1) + '.jpg')

        optimizer.zero_grad()

        mseLoss = 0
        ssimLoss = 0
        for i in range(len(map_list)):
            map1 = torch.from_numpy(map_list[i][0]).unsqueeze(0).cuda()
            map2 = torch.from_numpy(map_list[i][1]).unsqueeze(0).cuda()
            input1 = tensor_ir[i]
            input2 = tensor_vis[i]
            output = outputs[i]
            vsm_img = input1 * map1 + input2 * map2
            mseLoss += mse_loss(vsm_img, output)
            ssimLoss += 1 - ssim(vsm_img.unsqueeze(0), output.unsqueeze(0), normalize=True, val_range=1)

            # print(outputs[i])
            # print(vsm_img)
            # vsm_pil = tensor_to_pil(vsm_img.squeeze().squeeze())
            # vsm_pil.show()

        total_loss = mseLoss + ssimLoss
        total_loss.backward()

        # nn.utils.clip_grad_value_(model.parameters(), args.grad_clip)
        optimizer.step()
        logging.info("epoch: %d batch: %d total_loss: %f mse_loss: %f ssim_loss: %f ", epoch, batch, total_loss, mseLoss, ssimLoss)


if __name__ == '__main__':
    main()
