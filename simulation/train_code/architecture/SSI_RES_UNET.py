import math

import torch
import torch.nn as nn

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

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

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class Res2Net(nn.Module):
    def __init__(self, inChannel, uPlane, scale=4):
        super(Res2Net, self).__init__()
        self.uPlane = uPlane
        self.scale = scale

        self.conv_init = nn.Conv2d(inChannel, uPlane * scale, kernel_size=1, bias=False)
        self.bn_init = nn.BatchNorm2d(uPlane * scale)

        convs = []
        bns = []
        for i in range(self.scale - 1):
            convs.append(nn.Conv2d(self.uPlane, self.uPlane, kernel_size=3, stride=1, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(self.uPlane))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv_end = nn.Conv2d(uPlane * scale, inChannel, kernel_size=1, bias=False)
        self.bn_end = nn.BatchNorm2d(inChannel)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.conv_init(x)
        out = self.bn_init(out)
        out = self.relu(out)

        spx = torch.split(out, self.uPlane, 1)
        for i in range(self.scale - 1):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.scale - 1]), 1)

        out = self.conv_end(out)
        out = self.bn_end(out)
        return out

def conv_block(in_planes, out_planes, the_kernel=3, the_stride=1, the_padding=1, flag_norm=False, flag_norm_act=True):
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=the_kernel, stride=the_stride, padding=the_padding)
    activation = nn.ReLU(inplace=True)
    norm = nn.BatchNorm2d(out_planes)
    if flag_norm:
        return nn.Sequential(conv,norm,activation) if flag_norm_act else nn.Sequential(conv,activation,norm)
    else:
        return nn.Sequential(conv,activation)

def conv1x1_block(in_planes, out_planes, flag_norm=False):
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0,bias=False)
    norm = nn.BatchNorm2d(out_planes)
    return nn.Sequential(conv,norm) if flag_norm else conv

def fully_block(in_dim, out_dim, flag_norm=False, flag_norm_act=True):
    fc = nn.Linear(in_dim, out_dim)
    activation = nn.ReLU(inplace=True)
    norm = nn.BatchNorm2d(out_dim)
    if flag_norm:
        return nn.Sequential(fc,norm,activation) if flag_norm_act else nn.Sequential(fc,activation,norm)
    else:
        return nn.Sequential(fc,activation)

_NORM_BONE = False

class SSI_RES_UNET(nn.Module):

    def __init__(self, in_ch=28, out_ch=28, conv=default_conv):
        super(SSI_RES_UNET, self).__init__()

        n_resblocks = 16
        n_feats = 64
        kernel_size = 3
        scale = 1
        act = nn.ReLU(True)

        # define head module
        m_head = [conv(in_ch, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale= 1
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [conv(n_feats, out_ch, kernel_size)]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x, input_mask=None):
        # x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        # x = self.add_mean(x)

        return x
