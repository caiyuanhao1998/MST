import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
from pdb import set_trace as stx

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

def A(x,Phi):
    temp = x*Phi
    y = torch.sum(temp,1)
    return y

def At(y,Phi):
    temp = torch.unsqueeze(y, 1).repeat(1,Phi.shape[1],1,1)
    x = temp*Phi
    return x

def shift_3d(inputs,step=2):
    [bs, nC, row, col] = inputs.shape
    for i in range(nC):
        inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:], shifts=step*i, dims=2)
    return inputs

def shift_back_3d(inputs,step=2):
    [bs, nC, row, col] = inputs.shape
    for i in range(nC):
        inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:], shifts=(-1)*step*i, dims=2)
    return inputs

class Recon_block(nn.Module):
    def __init__(self, channelNum=31, filter_num=64):
        super(Recon_block, self).__init__()

        self.block_size = 48
        self.channel = channelNum
        self.scale = channelNum**-0.5

        # Local
        self.conv1 = nn.Conv2d(channelNum, filter_num, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(filter_num, channelNum, 3, 1, 1, bias=False)

        # Non-local
        self.conv3 = nn.Conv2d(channelNum, channelNum, 1, 1, 0, bias=False)

        #### activation function
        self.relu = nn.ReLU(inplace=True)

        self.cu = nn.Parameter(torch.zeros((1,channelNum,self.block_size,self.block_size)), requires_grad=True)

        self.wz1 = nn.Parameter(torch.zeros((1)), requires_grad=True)

    def forward(self, xt):
        b, c, h_inp, w_inp = xt.shape
        hb, wb = 48, 48
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        xt = F.pad(xt, [0, pad_w, 0, pad_h], mode='reflect')
        xt = rearrange(xt, 'b c (h b0) (w b1) -> (b h w) c b0 b1', b0=48, b1=48)
        b,c,h,w = xt.shape

        # Local
        x_resx1 = self.relu(self.conv1(xt))
        x_resx2 = self.conv2(x_resx1)
        z1 = xt + x_resx2

        # Non-local
        x_g = self.conv3(xt)
        q = xt.view(b,c,h*w).permute(0,2,1)
        kt = xt.view(b,c,h*w)
        v = x_g.view(b,c,h*w).permute(0,2,1)
        attn = torch.einsum('b m c, b c n -> b m n', q, kt)
        attn = torch.softmax(attn*self.scale, dim=-1)
        context = torch.einsum('b m n, b n c -> b m c', attn, v)
        context_softmax = context*(1.0/(self.block_size+self.channel-1)*self.block_size)
        context_softmax = context_softmax.view(b,c,h,w)

        z = self.wz1*z1 + (1-self.wz1)*self.relu(context_softmax)
        z = rearrange(z, '(b h w) c b0 b1 -> b c (h b0) (w b1)', h=(h_inp+pad_h)//hb, w=(w_inp+pad_w)//wb)

        return z[:,:,:h_inp,:w_inp]

class DNU(nn.Module):

    def __init__(self, channelNum=28, num_iterations=9):
        super(DNU, self).__init__()
        self.alphas = nn.Parameter(torch.zeros(num_iterations), requires_grad=True)+1e-8
        self.fution = nn.Conv2d(56, 28, 1, padding=0, bias=True)
        self.num_iterations = num_iterations
        self.denoisers = nn.ModuleList([])
        for _ in range(num_iterations):
            self.denoisers.append(
                Recon_block(channelNum),
            )

    def initial_z(self, y, Phi):
        """
        :param y: [b,256,310]
        :param Phi: [b,28,256,310]
        :return: z: [b,28,256,310]
        """
        nC, step = 28, 2
        y = y / nC * 2
        bs,row,col = y.shape
        z = torch.zeros(bs, nC, row, col).cuda().float()
        for i in range(nC):
            z[:, i, :, step * i:step * i + col - (nC - 1) * step] = y[:, :, step * i:step * i + col - (nC - 1) * step]
        z = self.fution(torch.cat([z, Phi], dim=1))
        return z

    def forward(self, y, input_mask=None):
        if input_mask == None:
            Phi = torch.rand((1, 28, 256, 310)).cuda()
            Phi_s = torch.rand((1, 256, 310)).cuda()
        else:
            Phi, Phi_s = input_mask

        # x = At(y, Phi)  # z0=H^T y  [1,28,256,310]
        x = self.initial_z(y, Phi)
        for i in range(self.num_iterations):
            alpha = self.alphas[i]
            z = self.denoisers[i](x)
            Phi_z = A(z, Phi)
            x = z + At(torch.div(y-Phi_z,alpha+Phi_s), Phi)
            # stx()

        return x[:, :, :, 0:256]

