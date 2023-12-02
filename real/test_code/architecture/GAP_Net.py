import torch.nn.functional as F
import torch
import torch.nn as nn

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

class double_conv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(double_conv, self).__init__()
        self.d_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.d_conv(x)
        return x


class Unet(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(Unet, self).__init__()

        self.dconv_down1 = double_conv(in_ch, 32)
        self.dconv_down2 = double_conv(32, 64)
        self.dconv_down3 = double_conv(64, 128)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            # nn.Conv2d(64, 64, (1,2), padding=(0,1)),
            nn.ReLU(inplace=True)
        )
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.dconv_up2 = double_conv(64 + 64, 64)
        self.dconv_up1 = double_conv(32 + 32, 32)

        self.conv_last = nn.Conv2d(32, out_ch, 1)
        self.afn_last = nn.Tanh()

    def forward(self, x):
        b, c, h_inp, w_inp = x.shape
        hb, wb = 8, 8
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
        inputs = x
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)

        x = self.upsample2(conv3)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample1(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        x = self.conv_last(x)
        x = self.afn_last(x)
        out = x + inputs

        return out[:, :, :h_inp, :w_inp]


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class GAP_net(nn.Module):

    def __init__(self):
        super(GAP_net, self).__init__()

        self.unet1 = Unet(28, 28)
        self.unet2 = Unet(28, 28)
        self.unet3 = Unet(28, 28)
        self.unet4 = Unet(28, 28)
        self.unet5 = Unet(28, 28)
        self.unet6 = Unet(28, 28)
        self.unet7 = Unet(28, 28)
        self.unet8 = Unet(28, 28)
        self.unet9 = Unet(28, 28)

    def forward(self, y, input_mask=None, input_mask_s=None):
        if input_mask==None:
            Phi = torch.rand((1,28,256,310)).cuda()
            Phi_s = torch.rand((1, 256, 310)).cuda()
        else:
            Phi, Phi_s = input_mask, input_mask_s
        x_list = []
        x = At(y, Phi)  # v0=H^T y
        ### 1-3
        yb = A(x, Phi)
        x = x + At(torch.div(y - yb, Phi_s), Phi)
        x = shift_back_3d(x)
        x = self.unet1(x)
        x = shift_3d(x)
        yb = A(x, Phi)
        x = x + At(torch.div(y - yb, Phi_s), Phi)
        x = shift_back_3d(x)
        x = self.unet2(x)
        x = shift_3d(x)
        yb = A(x, Phi)
        x = x + At(torch.div(y - yb, Phi_s), Phi)
        x = shift_back_3d(x)
        x = self.unet3(x)
        x = shift_3d(x)
        ### 4-6
        yb = A(x, Phi)
        x = x + At(torch.div(y - yb, Phi_s), Phi)
        x = shift_back_3d(x)
        x = self.unet4(x)
        x = shift_3d(x)
        yb = A(x, Phi)
        x = x + At(torch.div(y - yb, Phi_s), Phi)
        x = shift_back_3d(x)
        x = self.unet5(x)
        x = shift_3d(x)
        yb = A(x, Phi)
        x = x + At(torch.div(y - yb, Phi_s), Phi)
        x = shift_back_3d(x)
        x = self.unet6(x)
        x = shift_3d(x)
        # ### 7-9
        yb = A(x, Phi)
        x = x + At(torch.div(y - yb, Phi_s), Phi)
        x = shift_back_3d(x)
        x = self.unet7(x)
        x = shift_3d(x)
        x_list.append(x[:, :, :, 0:256])
        yb = A(x, Phi)
        x = x + At(torch.div(y - yb, Phi_s), Phi)
        x = shift_back_3d(x)
        x = self.unet8(x)
        x = shift_3d(x)
        x_list.append(x[:, :, :, 0:256])
        yb = A(x, Phi)
        x = x + At(torch.div(y - yb, Phi_s), Phi)
        x = shift_back_3d(x)
        x = self.unet9(x)
        x = shift_3d(x)
        return x[:, :, :, :384]