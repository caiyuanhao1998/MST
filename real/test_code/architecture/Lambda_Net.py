import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import einsum

class LambdaNetAttention(nn.Module):
    def __init__(
            self,
            dim,
    ):
        super().__init__()

        self.dim = dim
        self.to_q = nn.Linear(dim, dim//8, bias=False)
        self.to_k = nn.Linear(dim, dim//8, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.rescale = (dim//8)**-0.5
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0,2,3,1)
        b, h, w, c = x.shape

        # Reshape to (B,N,C), where N = window_size[0]*window_size[1] is the length of sentence
        x_inp = rearrange(x, 'b h w c -> b (h w) c')

        # produce query, key and value
        q = self.to_q(x_inp)
        k = self.to_k(x_inp)
        v = self.to_v(x_inp)

        # attention
        sim = einsum('b i d, b j d -> b i j', q, k)*self.rescale
        attn = sim.softmax(dim=-1)

        # aggregate
        out = einsum('b i j, b j d -> b i d', attn, v)

        # merge blocks back to original feature map
        out = rearrange(out, 'b (h w) c -> b h w c', h=h, w=w)
        out = self.gamma*out + x
        return out.permute(0,3,1,2)

class triple_conv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(triple_conv, self).__init__()
        self.t_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

    def forward(self, x):
        x = self.t_conv(x)
        return x

class double_conv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(double_conv, self).__init__()
        self.d_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

    def forward(self, x):
        x = self.d_conv(x)
        return x

def shift_back_3d(inputs,step=2):
    [bs, nC, row, col] = inputs.shape
    for i in range(nC):
        inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:], shifts=(-1)*step*i, dims=2)
    return inputs

class Lambda_Net(nn.Module):

    def __init__(self, out_ch=28):
        super(Lambda_Net, self).__init__()
        self.conv_in = nn.Conv2d(1+28, 28, 3, padding=1)

        # encoder
        self.conv_down1 = triple_conv(28, 32)
        self.conv_down2 = triple_conv(32, 64)
        self.conv_down3 = triple_conv(64, 128)
        self.conv_down4 = triple_conv(128, 256)
        self.conv_down5 = double_conv(256, 512)
        self.conv_down6 = double_conv(512, 1024)

        self.maxpool = nn.MaxPool2d(2)

        # decoder
        self.upsample5 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.upsample4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upsample3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upsample1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)

        self.conv_up1 = triple_conv(32+32, 32)
        self.conv_up2 = triple_conv(64+64, 64)
        self.conv_up3 = triple_conv(128+128, 128)
        self.conv_up4 = triple_conv(256+256, 256)
        self.conv_up5 = double_conv(512+512, 512)

        # attention
        self.attention = LambdaNetAttention(dim=128)
        
        self.conv_last1 = nn.Conv2d(32, 6, 3,1,1)
        self.conv_last2 = nn.Conv2d(38, 32, 3,1,1)
        self.conv_last3 = nn.Conv2d(32, 12, 3,1,1)
        self.conv_last4 = nn.Conv2d(44, 32, 3,1,1)
        self.conv_last5 = nn.Conv2d(32, out_ch, 1)
        self.act = nn.ReLU()

    def forward(self, x, input_mask=None):
        if input_mask == None:
            input_mask = torch.zeros((1,28,256,310)).cuda()
        x = x/28*2
        x = self.conv_in(torch.cat([x.unsqueeze(1), input_mask], dim=1))
        b, c, h_inp, w_inp = x.shape
        hb, wb = 32, 32
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
        res0 = x
        conv1 = self.conv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.conv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.conv_down3(x)
        x = self.maxpool(conv3)
        conv4 = self.conv_down4(x)
        x = self.maxpool(conv4)
        conv5 = self.conv_down5(x)
        x = self.maxpool(conv5)
        conv6 = self.conv_down6(x)

        x = self.upsample5(conv6)
        x = torch.cat([x, conv5], dim=1)
        x = self.conv_up5(x)

        x = self.upsample4(x)
        x = torch.cat([x, conv4], dim=1)
        x = self.conv_up4(x)

        x = self.upsample3(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.conv_up3(x)
        x = self.attention(x)

        x = self.upsample2(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample1(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.conv_up1(x)

        res1 = x
        out1 = self.act(self.conv_last1(x))
        x = self.conv_last2(torch.cat([res1,out1],dim=1))

        res2 = x
        out2 = self.act(self.conv_last3(x))
        out3 = self.conv_last4(torch.cat([res2, out2], dim=1))

        out = self.conv_last5(out3)+res0
        out = out[:, :, :h_inp, :w_inp]

        return shift_back_3d(out)[:, :, :, :256]


if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis
    import time
    inputs = torch.rand(1, 256, 310).cuda()
    model = Lambda_Net().cuda()
    flops = FlopCountAnalysis(model, inputs)
    n_param = sum([p.nelement() for p in model.parameters()])
    print(f'GMac:{flops.total()}')
    print(f'Params:{n_param}')
