import torch
import os
import argparse
from pathlib import Path
from utils import dataparallel
import scipy.io as sio
import numpy as np
from torch.autograd import Variable
from architecture import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

parser = argparse.ArgumentParser(description="PyTorch HSIFUSION")
parser.add_argument('--template', default='mst')
parser.add_argument('--gpu_id', default='0')
parser.add_argument('--data_path', default='../../datasets/TSA_real_data/Measurements/', type=str,help='path of data')
parser.add_argument('--mask_path', default='../../datasets/TSA_real_data/mask.mat', type=str,help='path of mask')
parser.add_argument("--size", default=660, type=int, help='the size of trainset image')
parser.add_argument("--seed", default=1, type=int, help='Random_seed')
parser.add_argument("--batch_size", default=1, type=int, help='batch_size')
parser.add_argument("--pretrained_model_path", default=None, type=str)
parser.add_argument("--outf", default='./exp/real_test_result/', type=str)
parser.add_argument("--method", type=str, default='others')
opt = parser.parse_args()
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
mask_3d_shift, mask_3d_shift_s = load_mask(opt.mask_path)

pretrained_model_path = opt.pretrained_model_path
save_path = opt.outf
if opt.method == 'others':
    opt.method = Path(pretrained_model_path).stem
model = model_generator(opt.method, opt.pretrained_model_path)
if isinstance(model, tuple):
    model = model[0]
model = model.cuda()
model = model.eval()
model = dataparallel(model, 1)
psnr_total = 0
k = 0
predictions = []
save_path = Path(save_path)
save_path.mkdir(parents=True, exist_ok=True)
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
    res = result.cpu().permute(2,3,1,0).squeeze(3).numpy()
    predictions.append(res)
    save_file = save_path / f'{j}.mat'
    sio.savemat(save_file, {'res':res})

sio.savemat(save_path / 'Test_result.mat', {'pred': np.stack(predictions)})
