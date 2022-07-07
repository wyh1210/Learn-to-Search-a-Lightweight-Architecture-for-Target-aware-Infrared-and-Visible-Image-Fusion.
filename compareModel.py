from model import DenseNet, FM
from ptflops import get_model_complexity_info, get_model_complexity_info2
import torch
import genotypes
import time
import os

with torch.cuda.device(0):
    # net = DenseNet()
    # macs, params = get_model_complexity_info(net, (2, 224, 224), as_strings=True,
    #                                          print_per_layer_stat=True, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # model_dir = r'C:\Users\ADMIN\Desktop\eff_net\trainTNORoadSceneSoftmax_0.5-20211004-191940'
    # model_path = os.path.join(model_dir, 'weights_epoch_50.pt')

    genotype_en1 = eval('genotypes.%s' % 'genotype1')
    genotype_en2 = eval('genotypes.%s' % 'genotype2')
    genotype_de = eval('genotypes.%s' % 'genotype3')

    model = FM(16, genotype_en1, genotype_en2, genotype_de).cuda()

    # params = torch.load(model_path)
    #
    # model.load_state_dict(params)
    #
    # model.eval()



    macs, params = get_model_complexity_info2(model, (1, 620, 448), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)

    # inputTensor = torch.ones(()).new_empty((1, 1, 620, 448)).cuda()
    # print(inputTensor.shape)
    # total = 0
    # for i in range(100):
    #     t1 = time.time()
    #     model(inputTensor, inputTensor)
    #     t2 = time.time()
    #     total += t2 - t1
    #
    # print(total/100)

    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))