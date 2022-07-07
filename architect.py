import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):

    def __init__(self, model, args, mse_loss, ssim_loss):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.mse = mse_loss
        self.ssim = ssim_loss
        para = [{'params': model.arch_parameters(), 'lr': args.arch_learning_rate}]
        self.optimizer = torch.optim.Adam(para,
                                          lr=args.arch_learning_rate, betas=(0.5, 0.999),
                                          weight_decay=args.arch_weight_decay)

    def step(self, input_valid):
        self.optimizer.zero_grad()
        self._backward_step(input_valid)
        self.optimizer.step()

    def _backward_step(self, valid_batch):
        tensor_ir, tensor_vis, map_list = valid_batch[0].cuda(), valid_batch[1].cuda(), valid_batch[2]
        outputs, lat = self.model(tensor_vis, tensor_ir)
        mseLoss = 0
        ssimLoss = 0
        for i in range(len(map_list)):
            map1 = torch.from_numpy(map_list[i][0]).unsqueeze(0).cuda()
            map2 = torch.from_numpy(map_list[i][1]).unsqueeze(0).cuda()
            input1 = tensor_ir[i]

            input2 = tensor_vis[i]
            output = outputs[i]
            vsm_img = input1 * map1 + input2 * map2

            mseLoss += self.mse(vsm_img, output)
            ssimLoss += 1 - self.ssim(vsm_img.unsqueeze(0), output.unsqueeze(0), normalize=True, val_range=1)

        total_loss = mseLoss + ssimLoss*10 + 0.05*lat
        print('mse: {} lat: {}'.format(mseLoss, lat))
        total_loss.backward()


