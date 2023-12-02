import argparse

parser = argparse.ArgumentParser(description="HyperSpectral Image Reconstruction Toolbox")

# Hardware specifications
parser.add_argument("--gpu_id", type=str, default='0')

# Data specifications
parser.add_argument('--data_root', type=str, default='../../datasets/', help='dataset directory')
parser.add_argument('--data_path_CAVE', default='../../datasets/CAVE_512_28/', type=str,
                        help='path of data')
parser.add_argument('--data_path_KAIST', default='../../datasets/KAIST_CVPR2021/', type=str,
                    help='path of data')
parser.add_argument('--mask_path', default='../../datasets/TSA_real_data/mask.mat', type=str,
                    help='path of mask')

# Saving specifications
parser.add_argument('--outf', type=str, default='./exp/bisrnet/', help='saving_path')

# Model specifications
parser.add_argument('--method', type=str, default='bisrnet', help='method name')
parser.add_argument('--pretrained_model_path', type=str, default=None, help='pretrained model directory')
parser.add_argument("--input_setting", type=str, default='H',
                    help='the input measurement of the network: H, HM or Y')
parser.add_argument("--input_mask", type=str, default='Phi',
                    help='the input mask of the network: Phi, Phi_PhiPhiT, Mask or None')  # Phi: shift_mask   Mask: mask


# Training specifications
parser.add_argument("--size", default=96, type=int, help='cropped patch size')
parser.add_argument("--epoch_sam_num", default=5000, type=int, help='total number of trainset')
parser.add_argument("--seed", default=1, type=int, help='Random_seed')
parser.add_argument('--batch_size', type=int, default=8, help='the number of HSIs per batch')
parser.add_argument("--isTrain", default=True, type=bool, help='train or test')
parser.add_argument("--max_epoch", type=int, default=500, help='total epoch')
parser.add_argument("--scheduler", type=str, default='MultiStepLR', help='MultiStepLR or CosineAnnealingLR')
parser.add_argument("--milestones", type=int, default=[50,100,150,200,250], help='milestones for MultiStepLR')
parser.add_argument("--gamma", type=float, default=0.1, help='learning rate decay for MultiStepLR')
parser.add_argument("--learning_rate", type=float, default=0.0001)

opt = parser.parse_args()
opt.input_setting = 'H'
opt.input_mask = 'Mask'
opt.scheduler = 'CosineAnnealingLR'

opt.trainset_num = 20000 // ((opt.size // 96) ** 2)

for arg in vars(opt):
    if vars(opt)[arg] == 'True':
        vars(opt)[arg] = True
    elif vars(opt)[arg] == 'False':
        vars(opt)[arg] = False
