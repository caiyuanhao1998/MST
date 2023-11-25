import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
import os
from pdb import set_trace as stx

# --------------------------------------------- Binarized Basic Units -----------------------------------------------------------------


class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        # stx()
        out = x + self.bias.expand_as(x)
        return out

class ReDistribution(nn.Module):
    def __init__(self, out_chn):
        super(ReDistribution, self).__init__()
        self.b = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)
        self.k = nn.Parameter(torch.ones(1,out_chn,1,1), requires_grad=True)
    
    def forward(self, x):
        out = x * self.k.expand_as(x) + self.b.expand_as(x)
        return out

class RPReLU(nn.Module):
    def __init__(self, inplanes):
        super(RPReLU, self).__init__()
        self.pr_bias0 = LearnableBias(inplanes)
        self.pr_prelu = nn.PReLU(inplanes)
        self.pr_bias1 = LearnableBias(inplanes)

    def forward(self, x):
        x = self.pr_bias1(self.pr_prelu(self.pr_bias0(x)))      
        return x

class Spectral_Binary_Activation(nn.Module):
    def __init__(self):
        super(Spectral_Binary_Activation, self).__init__()
        self.beta = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):

        binary_activation_no_grad = torch.sign(x)
        tanh_activation = torch.tanh(x*self.beta)
        
        out = binary_activation_no_grad.detach() - tanh_activation.detach() + tanh_activation
        return out


class HardBinaryConv(nn.Conv2d):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1, groups=1, bias=True):
        super(HardBinaryConv, self).__init__(
            in_chn,
            out_chn,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias
        )

    def forward(self, x):
        real_weights = self.weight
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        scaling_factor = scaling_factor.detach()
        # stx()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        # stx()
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        y = F.conv2d(x, binary_weights,self.bias, stride=self.stride, padding=self.padding, groups=self.groups)

        return y

class BinaryConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, groups=1):
        super(BinaryConv2d, self).__init__()

        self.move0 = ReDistribution(in_channels)
        self.binary_activation = Spectral_Binary_Activation()
        self.binary_conv = HardBinaryConv(in_chn=in_channels,
        out_chn=in_channels,
        kernel_size=kernel_size,
        stride = stride,
        padding=padding,
        bias=bias,
        groups=groups)
        self.relu=RPReLU(in_channels)


    def forward(self, x):
        out = self.move0(x)
        out = self.binary_activation(out)
        out = self.binary_conv(out)
        out =self.relu(out)
        out = out + x
        return out


class BinaryConv2d_Down(nn.Module):
    '''
    input: b,c,h,w
    output: b,c/2,2h,2w
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, groups=1):
        super(BinaryConv2d_Down, self).__init__()

        self.biconv_1 = BinaryConv2d(in_channels, in_channels, kernel_size, stride, padding, bias, groups)
        self.biconv_2 = BinaryConv2d(in_channels, in_channels, kernel_size, stride, padding, bias, groups)
        self.avg_pool = nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0)

    def forward(self, x):
        '''
        x: b,c,h,w
        out: b,2c,h/2,w/2
        '''
        out = self.avg_pool(x)
        out_1 = out
        out_2 = out_1.clone()
        out_1 = self.biconv_1(out_1)
        out_2 = self.biconv_2(out_2)
        out = torch.cat([out_1, out_2], dim=1)

        return out



class BinaryConv2d_Up(nn.Module):
    '''
    input: b,c,h,w
    output: b,c/2,2h,2w
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, groups=1):
        super(BinaryConv2d_Up, self).__init__()

        self.biconv_1 = BinaryConv2d(out_channels, out_channels, kernel_size, stride, padding, bias, groups)
        self.biconv_2 = BinaryConv2d(out_channels, out_channels, kernel_size, stride, padding, bias, groups)

    def forward(self, x):
        '''
        x: b,c,h,w
        out: b,c/2,2h,2w
        '''
        b,c,h,w = x.shape
        out = F.interpolate(x, scale_factor=2, mode='bilinear')
        
        out_1 = out[:,:c//2,:,:]
        out_2 = out[:,c//2:,:,:]

        out_1 = self.biconv_1(out_1)
        out_2 = self.biconv_2(out_2)

        
        out = (out_1 + out_2) / 2

        return out


class BinaryConv2d_Fusion_Decrease(nn.Module):
    '''
    input: b,c,h,w
    output: b,c/2,h,w
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, groups=1):
        super(BinaryConv2d_Fusion_Decrease, self).__init__()

        self.biconv_1 = BinaryConv2d(out_channels, out_channels, kernel_size, stride, padding, bias, groups)
        self.biconv_2 = BinaryConv2d(out_channels, out_channels, kernel_size, stride, padding, bias, groups)
        # self.avg_pool = nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0)

    def forward(self, x):
        '''
        x: b,c,h,w
        out: b,c/2,h,w
        '''
        b,c,h,w = x.shape
        out = x
        
        out_1 = out[:,:c//2,:,:]
        out_2 = out[:,c//2:,:,:]

        out_1 = self.biconv_1(out_1)
        out_2 = self.biconv_2(out_2)
        
        out = (out_1 + out_2) / 2

        return out


class BinaryConv2d_Fusion_Increase(nn.Module):
    '''
    input: b,c,h,w
    output: b,2c,h,w
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, groups=1):
        super(BinaryConv2d_Fusion_Increase, self).__init__()

        self.biconv_1 = BinaryConv2d(in_channels, in_channels, kernel_size, stride, padding, bias, groups)
        self.biconv_2 = BinaryConv2d(in_channels, in_channels, kernel_size, stride, padding, bias, groups)
        # self.avg_pool = nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0)

    def forward(self, x):
        '''
        x: b,c,h,w
        out: b,2c,h,w
        '''
        # stx()
        out_1 = x
        out_2 = out_1.clone()
        out_1 = self.biconv_1(out_1)
        out_2 = self.biconv_2(out_2)
        out = torch.cat([out_1, out_2], dim=1)

        return out

# ---------------------------------------------------------- Binarized UNet------------------------------------------------------



class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


def shift_back(inputs,step=2):          # input [bs,28,256,310]  output [bs, 28, 256, 256]
    [bs, nC, row, col] = inputs.shape
    down_sample = 256//row
    step = float(step)/float(down_sample*down_sample)
    out_col = row
    for i in range(nC):
        inputs[:,i,:,:out_col] = \
            inputs[:,i,:,int(step*i):int(step*i)+out_col]
    return inputs[:, :, :, :out_col]



class FeedForward(nn.Module):
    def __init__(self, dim, mult=2):
        super().__init__()
        self.net = nn.Sequential(
            BinaryConv2d_Fusion_Increase(dim, dim * mult, 1, 1, bias=False),
            BinaryConv2d_Fusion_Increase(dim * mult, dim * mult * mult, 1, 1, bias=False),
            RPReLU(dim * mult * mult),
            BinaryConv2d(dim * mult * mult, dim * mult * mult, 3, 1, 1, bias=False, groups=dim),
            RPReLU(dim * mult * mult),
            BinaryConv2d_Fusion_Decrease(dim * mult * mult, dim * mult, 1, 1, bias=False),
            BinaryConv2d_Fusion_Decrease(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)



class BiSRNet_Block(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
            num_blocks,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(
                PreNorm(dim, FeedForward(dim=dim))
            )

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for ff in self.blocks:
            # x = attn(x) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out



class BiSRNet_body(nn.Module):
    def __init__(self, in_dim=28, out_dim=28, dim=28, stage=2, num_blocks=[2,4,4]):
        super(BiSRNet_body, self).__init__()
        self.dim = dim
        self.stage = stage

        # Input projection
        self.embedding = BinaryConv2d(in_dim, self.dim, 3, 1, 1, bias=False)                           # 1-bit -> 32-bit

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                BiSRNet_Block(dim=dim_stage, num_blocks=num_blocks[i], dim_head=dim, heads=dim_stage // dim),
                BinaryConv2d_Down(dim_stage, dim_stage * 2, 3, 1, 1, bias=False),
            ]))
            dim_stage *= 2

        # Bottleneck
        self.bottleneck = BiSRNet_Block(
            dim=dim_stage, dim_head=dim, heads=dim_stage // dim, num_blocks=num_blocks[-1])

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(stage):
            self.decoder_layers.append(nn.ModuleList([
                BinaryConv2d_Up(dim_stage, dim_stage // 2, 3, 1, 1, bias=False),
                BinaryConv2d_Fusion_Decrease(dim_stage, dim_stage // 2, 1, 1, bias=False),
                BiSRNet_Block(
                    dim=dim_stage // 2, num_blocks=num_blocks[stage - 1 - i], dim_head=dim,
                    heads=(dim_stage // 2) // dim),
            ]))
            dim_stage //= 2

        # Output projection
        self.mapping = BinaryConv2d(self.dim, out_dim, 3, 1, 1, bias=False)                                # 1-bit -> 32-bit

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """

        # Embedding
        fea = self.embedding(x)

        # Encoder
        fea_encoder = []
        for (BiSRNet_Block, FeaDownSample) in self.encoder_layers:
            # stx()
            fea = BiSRNet_Block(fea)
            fea_encoder.append(fea)
            # stx()
            fea = FeaDownSample(fea)

        # Bottleneck
        fea = self.bottleneck(fea)

        # Decoder
        for i, (FeaUpSample, Fution, BiSRNet_Block) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(torch.cat([fea, fea_encoder[self.stage-1-i]], dim=1))
            fea = BiSRNet_Block(fea)

        # Mapping
        out = self.mapping(fea) + x

        return out



class BiSRNet(nn.Module):
    '''
    Only 3 layers are 32-bit conv
    '''
    def __init__(self, in_channels=28, out_channels=28, n_feat=28, stage=3, num_blocks=[1,1,1]):
        super(BiSRNet, self).__init__()
        self.stage = stage
        self.conv_in = nn.Conv2d(in_channels, n_feat, kernel_size=3, padding=(3 - 1) // 2,bias=False)       # 1-bit -> 32-bit
        modules_body = [BiSRNet_body(dim=n_feat, stage=2, num_blocks=num_blocks) for _ in range(stage)]
        self.fution = nn.Conv2d(56, 28, 1, padding=0, bias=True)                                            # 1-bit -> 32-bit
        self.body = nn.Sequential(*modules_body)
        self.conv_out = nn.Conv2d(n_feat, out_channels, kernel_size=3, padding=(3 - 1) // 2,bias=False)     # 1-bit -> 32-bit

    def initial_x(self, y, Phi):
        """
        :param y: [b,256,310]
        :param Phi: [b,28,256,256]
        :return: z: [b,28,256,256]
        """
        x = self.fution(torch.cat([y, Phi], dim=1))
        return x

    def forward(self, y, Phi=None):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        if Phi==None:
            Phi = torch.rand((1,28,256,256)).cuda()
            # Phi = torch.rand((1,28,256,256)).to(device)
        x = self.initial_x(y, Phi)
        b, c, h_inp, w_inp = x.shape
        hb, wb = 8, 8
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
        x = self.conv_in(x)
        h = self.body(x)
        h = self.conv_out(h)
        h += x
        return h[:, :, :h_inp, :w_inp]