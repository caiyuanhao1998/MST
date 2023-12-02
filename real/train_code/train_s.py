from architecture import *
from utils import *
from dataset import dataset
import torch.utils.data as tud
import torch
import torch.nn.functional as F
import time
import datetime
from torch.autograd import Variable
import os
from option_s import opt

from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from pdb import set_trace as stx

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

if not torch.cuda.is_available():
    raise Exception('NO GPU!')

# load training data
CAVE = prepare_data_cave(opt.data_path_CAVE, 30)
KAIST = prepare_data_KAIST(opt.data_path_KAIST, 30)
# KAIST = None

# saving path
date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
opt.outf = os.path.join(opt.outf, date_time)
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

model_path = opt.outf  
# model
model = model_generator(opt.method, opt.pretrained_model_path).cuda()

#optimizing
optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999))
if opt.scheduler == 'MultiStepLR':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.gamma)
elif opt.scheduler == 'CosineAnnealingLR':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.max_epoch, eta_min=1e-6)

criterion = nn.L1Loss()


if __name__ == "__main__":

    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    ## Initialize weight
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.xavier_uniform_(layer.weight)
            # nn.init.constant_(layer.bias, 0.0)
        if isinstance(layer, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(layer.weight)
            # nn.init.constant_(layer.bias, 0.0)

    logger = gen_log(model_path)
    logger.info("Learning rate:{}, batch_size:{}.\n".format(opt.learning_rate, opt.batch_size))

    ## pipline of training
    for epoch in range(0, opt.max_epoch):
        model.train()
        
        Dataset = dataset(opt, CAVE, KAIST)
        # Dataset = dataset(opt, CAVE)
        loader_train = tud.DataLoader(Dataset, num_workers=8, batch_size=opt.batch_size, shuffle=True)

        scheduler.step(epoch)
        epoch_loss = 0

        start_time = time.time()
        for i, (input, label, Mask, Phi, Phi_s) in enumerate(loader_train):
            input, label = Variable(input), Variable(label)
            input, label = input.cuda(), label.cuda()
            out = model(input)

            loss = criterion(out, label)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            if i % (1000) == 0:
                print('%4d %4d / %4d loss = %.10f time = %s' % (
                    epoch + 1, i, len(Dataset) // opt.batch_size, epoch_loss / ((i + 1) * opt.batch_size),
                    datetime.datetime.now()))

        elapsed_time = time.time() - start_time

        logger.info("===> Epoch {} Complete: Avg. Loss: {:.6f} time: {:.2f} s".
                    format(epoch+1 , epoch_loss / len(Dataset), elapsed_time))
        
        print('epcoh = %4d , loss = %.10f , time = %4.2f s' % (epoch + 1, epoch_loss / len(Dataset), elapsed_time))


        torch.save(model, os.path.join(opt.outf, 'model_%03d.pth' % (epoch + 1)))


