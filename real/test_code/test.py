import torch
import os
import argparse
from utils import dataparallel
import scipy.io as sio
import numpy as np
from torch.autograd import Variable

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="PyTorch HSIFUSION")
parser.add_argument('--data_path', default='./Data/Testing_data/', type=str,help='path of data')
parser.add_argument('--mask_path', default='./Data/mask.mat', type=str,help='path of mask')
parser.add_argument("--size", default=660, type=int, help='the size of trainset image')
parser.add_argument("--trainset_num", default=2000, type=int, help='total number of trainset')
parser.add_argument("--testset_num", default=5, type=int, help='total number of testset')
parser.add_argument("--seed", default=1, type=int, help='Random_seed')
parser.add_argument("--batch_size", default=1, type=int, help='batch_size')
parser.add_argument("--isTrain", default=False, type=bool, help='train or test')
parser.add_argument("--pretrained_model_path", default=None, type=str)
opt = parser.parse_args()
print(opt)

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

def load_mask(path,size=660):
    ## load mask
    data = sio.loadmat(path)
    mask = data['mask']
    mask_3d = np.tile(mask[:, :, np.newaxis], (1, 1, 28))
    mask_3d_shift = np.zeros((size, size + (28 - 1) * 2, 28))
    mask_3d_shift[:, 0:size, :] = mask_3d
    for t in range(28):
        mask_3d_shift[:, :, t] = np.roll(mask_3d_shift[:, :, t], 2 * t, axis=1)
    mask_3d_shift_s = np.sum(mask_3d_shift ** 2, axis=2, keepdims=False)
    mask_3d_shift_s[mask_3d_shift_s == 0] = 1
    mask_3d_shift = torch.FloatTensor(mask_3d_shift.copy()).permute(2, 0, 1)
    mask_3d_shift_s = torch.FloatTensor(mask_3d_shift_s.copy())
    return mask_3d_shift.unsqueeze(0), mask_3d_shift_s.unsqueeze(0)

HR_HSI = prepare_data(opt.data_path, 5)
mask_3d_shift, mask_3d_shift_s = load_mask('./Data/mask.mat')

pretrained_model_path = "/data/lj/exp/hsi/nips2022/dgsmp_real_exp/exp8/hdnet_p384_b1_cosine/2022_05_13_23_05_15/model_150.pth"
model = torch.load(pretrained_model_path)
model = model.eval()
model = dataparallel(model, 1)
psnr_total = 0
k = 0
for j in range(5):
    with torch.no_grad():
        meas = HR_HSI[:,:,j]
        meas = meas / meas.max() * 0.8
        meas = torch.FloatTensor(meas)
        # meas = torch.FloatTensor(meas).unsqueeze(2).permute(2, 0, 1)
        input = meas.unsqueeze(0)
        input = Variable(input)
        input = input.cuda()
        mask_3d_shift = mask_3d_shift.cuda()
        mask_3d_shift_s = mask_3d_shift_s.cuda()
        out = model(input, mask_3d_shift, mask_3d_shift_s)
        result = out
        result = result.clamp(min=0., max=1.)
    k = k + 1
    if not os.path.exists(save_path):  # Create the model directory if it doesn't exist
        os.makedirs(save_path)
    res = result.cpu().permute(2,3,1,0).squeeze(3).numpy()
    save_file = save_path + f'{j}.mat'
    sio.savemat(save_file, {'res':res})
