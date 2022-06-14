from architecture import *
from utils import *
import scipy.io as scio
import torch
import os
import numpy as np
from option import opt

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')

# Intialize mask
mask3d_batch, input_mask = init_mask(opt.mask_path, opt.input_mask, 10)

if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

def test(model):
    test_data = LoadTest(opt.test_path)
    test_gt = test_data.cuda().float()
    input_meas = init_meas(test_gt, mask3d_batch, opt.input_setting)
    model.eval()
    with torch.no_grad():
        model_out = model(input_meas, input_mask)
    pred = np.transpose(model_out.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    truth = np.transpose(test_gt.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    model.train()
    return pred, truth

def main():
    # model
    if opt.method == 'hdnet':
        model, FDL_loss = model_generator(opt.method, opt.pretrained_model_path)
        model = model.cuda()
    else:
        model = model_generator(opt.method, opt.pretrained_model_path).cuda()
    pred, truth = test(model)
    name = opt.outf + 'Test_result.mat'
    print(f'Save reconstructed HSIs as {name}.')
    scio.savemat(name, {'truth': truth, 'pred': pred})

if __name__ == '__main__':
    main()