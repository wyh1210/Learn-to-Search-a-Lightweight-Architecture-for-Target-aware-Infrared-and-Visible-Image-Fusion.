import os
import random
import sys
import time
import glob
import numpy as np
import torch
from PIL import Image
from os import listdir
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
# from model_search import Network
from model_search2 import FM
from architect import Architect
import pytorch_msssim
import torchvision.transforms as transforms
import pickle
parser = argparse.ArgumentParser("untitled")

parser.add_argument('--batch_size', type=int, default=4, help='batch size')  # 64改成了4
parser.add_argument('--learning_rate', type=float, default=1e-4, help='init learning rate')  #0.025-->2e-4
parser.add_argument('--learning_rate_min', type=float, default=1e-5, help='min learning rate')  #0.001-->1e-4
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--is_cuda', type=bool, default=True)
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=2e-4, help='learning rate for arch encoding')  # 3e-4
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')  # 1e-3
parser.add_argument('--dataset1', type=str, default=r'C:\Users\ADMIN\Desktop\roadSceneTNO128\crop_infrared128_20', help='Infrared images for training')
parser.add_argument('--dataset2', type=str, default=r'C:\Users\ADMIN\Desktop\roadSceneTNO128\crop_visible128_20', help='Visible images for training')
args = parser.parse_args()
args.save = 'search-B{}-{}'.format(args.batch_size, time.strftime("%Y%m%d-%H%M%S"))
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
    ssim_loss = pytorch_msssim.msssim

    # if args.is_cuda:
    #     with open(r'./operations2/latency_gpu_en.pkl', 'rb') as f:
    #         latency_en = pickle.load(f)
    #     with open(r'./operations2/latency_gpu_de.pkl', 'rb') as f:
    #         latency_de = pickle.load(f)
    #     model = Network(latency_en, latency_de).cuda()
    with open(r'./latency_gpu.pkl', 'rb') as f:
        latency = pickle.load(f)
    model = FM(16, latency).cuda()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    para = [{'params': model.parameters(), 'lr': args.learning_rate}]
    optimizer = torch.optim.SGD(
        para,
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
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

    train_num = len(Infrared_path_list)//2

    trainQueue = imgQueue[:train_num]
    validQueue = imgQueue[train_num:train_num*2]

    train_queue, batches = utils.load_dataset(trainQueue, args.batch_size)
    valid_queue, batches = utils.load_dataset(validQueue, args.batch_size)

    # print(train_queue[0])
    # print(valid_queue[0])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.learning_rate_min)
    architect = Architect(model, args, mse_loss, ssim_loss)

    for epoch in range(epochs):
        # lr = scheduler.get_lr()[0]
        lr = scheduler.get_last_lr()

        logging.info('epoch %d lr %e', epoch, lr[0])

        # train(train_queue, valid_queue, batches, model, architect, mse_loss, ssim_loss, optimizer, lr, epoch)

        genotype1, genotype2, genotype3 = model.genotype()
        logging.info('genotype1 = %s', genotype1)
        logging.info('genotype2 = %s', genotype2)
        logging.info('genotype3 = %s', genotype3)

        # print(F.softmax(model.alphas1, dim=-1))
        # print(F.softmax(model.alphas2, dim=-1))
        # print(F.softmax(model.alphas3, dim=-1))
        logging.info(F.softmax(model.alphas1, dim=-1).cpu().item().numpy())
        logging.info(F.softmax(model.alphas2, dim=-1).cpu().item().numpy())
        logging.info(F.softmax(model.alphas3, dim=-1).cpu().item().numpy())

        if (epoch+1) % 5 == 0:
            utils.save(model, os.path.join(args.save, 'weights_epoch'+str(epoch+1)+'.pt'))
        scheduler.step()


tensor_to_pil = transforms.ToPILImage()


def train(train_queue, valid_queue, batches, model, architect, mse_loss, ssim, optimizer, lr, epoch):
    # model.train()
    for batch in range(batches):
        paths_train = train_queue[batch * args.batch_size:(batch * args.batch_size + args.batch_size)]  # 训练一批
        train_batch = utils.get_batch(paths_train)
        # print(train_batch[0].shape, train_batch[1].shape, train_batch[2][0])

        paths_valid = valid_queue[batch * args.batch_size:(batch * args.batch_size + args.batch_size)]  # 训练一批
        valid_batch = utils.get_batch(paths_valid)  # 取出一批图片并且变成张量

        architect.step(valid_batch)

        print(F.softmax(model.alphas1, dim=-1))
        print(F.softmax(model.alphas2, dim=-1))
        print(F.softmax(model.alphas3, dim=-1))

        tensor_ir, tensor_vis, map_list = train_batch[0].cuda(), train_batch[1].cuda(), train_batch[2]
        outputs, _ = model(tensor_ir, tensor_vis)

        if epoch>-1 and (batch + 1) % 100 == 0:
            output_tmp = outputs.detach().cpu()
            # tensor = torch.squeeze(output_tmp[0])
            tensor = torch.squeeze(output_tmp[0][0])
            # array = np.asarray(tensor)
            # image = Image.fromarray(array)
            # image = image.convert('L')
            # print(tensor)
            image = tensor_to_pil(tensor)
            image.save(args.save+'/output/image_batch'+str(batch+1)+'.jpg')

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
        mseLoss = mseLoss/len(map_list)
        ssimLoss = ssimLoss/len(map_list)
        total_loss = mseLoss + ssimLoss
        total_loss.backward()

        # nn.utils.clip_grad_value_(model.parameters(), args.grad_clip)
        optimizer.step()
        logging.info("epoch: %d batch: %d loss: %f", epoch, batch+1, total_loss)


if __name__ == '__main__':
    main()
