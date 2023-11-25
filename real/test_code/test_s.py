import torch
import os
import argparse

import scipy.io as sio
import numpy as np
from torch.autograd import Variable
from architecture import *


parser = argparse.ArgumentParser(description="PyTorch HSIFUSION")
parser.add_argument('--data_path', default='../../datasets/TSA_real_data/Measurements/', type=str,help='path of data')
parser.add_argument('--mask_path', default='../../datasets/TSA_real_data/mask.mat', type=str,help='path of mask')
parser.add_argument("--size", default=660, type=int, help='the size of trainset image')
parser.add_argument("--trainset_num", default=2000, type=int, help='total number of trainset')
parser.add_argument("--testset_num", default=5, type=int, help='total number of testset')
parser.add_argument("--seed", default=1, type=int, help='Random_seed')
parser.add_argument("--batch_size", default=1, type=int, help='batch_size')
parser.add_argument("--isTrain", default=False, type=bool, help='train or test')
parser.add_argument("--gpu_id", type=str, default='0')
parser.add_argument("--pretrained_model_path", type=str, default='model_zoo/bisrnet/bisrnet.pth')
parser.add_argument("--outf", type=str, default='./exp/bisrnet/')
parser.add_argument("--method", type=str, default='bisrnet')
opt = parser.parse_args()
print(opt)

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

def prepare_data(path, file_num):
    HR_HSI = np.zeros((((660,714,file_num))))
    for idx in range(file_num):
        ####  read HrHSI
        path1 = os.path.join(path) + 'scene' + str(idx+1) + '.mat'
        data = sio.loadmat(path1)
        HR_HSI[:,:,idx] = data['meas_real']
        HR_HSI[HR_HSI < 0] = 0.0
        HR_HSI[HR_HSI > 1] = 1.0
    return HR_HSI



HR_HSI = prepare_data(opt.data_path, 5)

model = model_generator(opt.method, opt.pretrained_model_path).cuda()
model = model.eval()
psnr_total = 0
k = 0
for j in range(5):
    with torch.no_grad():
        meas = HR_HSI[:,:,j]
        meas = meas / meas.max() * 0.8
        meas = torch.FloatTensor(meas).unsqueeze(2).permute(2, 0, 1)
        input = meas.unsqueeze(0)
        input = Variable(input)
        input = input.cuda()
        out = model(input)
        result = out
        result = result.clamp(min=0., max=1.)
    k = k + 1
    model_dir = opt.outf
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)
    res = result.cpu().permute(2,3,1,0).squeeze(3).numpy()
    save_path = model_dir + str(j + 1) + '.mat'
    sio.savemat(save_path, {'res':res})



