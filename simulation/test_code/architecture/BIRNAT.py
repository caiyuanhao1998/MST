import torch
import torch.nn as nn
import torch.nn.functional as F

class self_attention(nn.Module):
    def __init__(self, ch):
        super(self_attention, self).__init__()
        self.conv1 = nn.Conv2d(ch, ch // 8, 1)
        self.conv2 = nn.Conv2d(ch, ch // 8, 1)
        self.conv3 = nn.Conv2d(ch, ch, 1)
        self.conv4 = nn.Conv2d(ch, ch, 1)
        self.gamma1 = torch.nn.Parameter(torch.Tensor([0]))
        self.ch = ch

    def forward(self, x):
        batch_size = x.shape[0]

        f = self.conv1(x)
        g = self.conv2(x)
        h = self.conv3(x)
        ht = h.reshape([batch_size, self.ch, -1])

        ft = f.reshape([batch_size, self.ch // 8, -1])
        n = torch.matmul(ft.permute([0, 2, 1]), g.reshape([batch_size, self.ch // 8, -1]))
        beta = F.softmax(n, dim=1)

        o = torch.matmul(ht, beta)
        o = o.reshape(x.shape)  # [bs, C, h, w]

        o = self.conv4(o)

        x = self.gamma1 * o + x

        return x


class res_part(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(res_part, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            # nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            # nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            # nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x = x1 + x
        x1 = self.conv2(x)
        x = x1 + x
        x1 = self.conv3(x)
        x = x1 + x
        return x


class down_feature(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(down_feature, self).__init__()
        self.conv = nn.Sequential(
            # nn.Conv2d(in_ch, 20, 5, stride=1, padding=2),
            # nn.Conv2d(20, 40, 5, stride=2, padding=2),
            # nn.Conv2d(40, out_ch, 5, stride=2, padding=2),
            nn.Conv2d(in_ch, 20, 5, stride=1, padding=2),
            nn.Conv2d(20, 20, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(20, 20, 3, stride=1, padding=1),
            nn.Conv2d(20, 40, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(40, out_ch, 3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_feature(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(up_feature, self).__init__()
        self.conv = nn.Sequential(
            # nn.ConvTranspose2d(in_ch, 40, 3, stride=2, padding=1, output_padding=1),
            # nn.ConvTranspose2d(40, 20, 3, stride=2, padding=1, output_padding=1),
            nn.Conv2d(in_ch, 40, 3, stride=1, padding=1),
            nn.Conv2d(40, 30, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(30, 20, 3, stride=1, padding=1),
            nn.Conv2d(20, 20, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(20, 20, 3, padding=1),
            nn.Conv2d(20, out_ch, 1),
            # nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class cnn1(nn.Module):
    # 输入meas concat mask
    # 3 下采样

    def __init__(self, B):
        super(cnn1, self).__init__()
        self.conv1 = nn.Conv2d(B + 1, 32, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=1, stride=1)
        self.relu3 = nn.LeakyReLU(inplace=True)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.relu4 = nn.LeakyReLU(inplace=True)
        self.conv5 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu5 = nn.LeakyReLU(inplace=True)
        self.conv51 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.relu51 = nn.LeakyReLU(inplace=True)
        self.conv52 = nn.Conv2d(32, 16, kernel_size=1, stride=1)
        self.relu52 = nn.LeakyReLU(inplace=True)
        self.conv6 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        self.res_part1 = res_part(128, 128)
        self.res_part2 = res_part(128, 128)
        self.res_part3 = res_part(128, 128)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.relu7 = nn.LeakyReLU(inplace=True)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=1, stride=1)
        self.conv9 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.relu9 = nn.LeakyReLU(inplace=True)
        self.conv10 = nn.Conv2d(128, 128, kernel_size=1, stride=1)

        self.att1 = self_attention(128)

    def forward(self, meas=None, nor_meas=None, PhiTy=None):
        data = torch.cat([torch.unsqueeze(nor_meas, dim=1), PhiTy], dim=1)
        out = self.conv1(data)

        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.conv4(out)
        out = self.relu4(out)
        out = self.res_part1(out)
        out = self.conv7(out)
        out = self.relu7(out)
        out = self.conv8(out)
        out = self.res_part2(out)
        out = self.conv9(out)
        out = self.relu9(out)
        out = self.conv10(out)
        out = self.res_part3(out)

        # out = self.att1(out)

        out = self.conv5(out)
        out = self.relu5(out)
        out = self.conv51(out)
        out = self.relu51(out)
        out = self.conv52(out)
        out = self.relu52(out)
        out = self.conv6(out)

        return out


class forward_rnn(nn.Module):

    def __init__(self):
        super(forward_rnn, self).__init__()
        self.extract_feature1 = down_feature(1, 20)
        self.up_feature1 = up_feature(60, 1)
        self.conv_x1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=1, padding=2),
            nn.Conv2d(16, 32, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 40, 3, stride=2, padding=1),
            nn.Conv2d(40, 20, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(20, 20, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(20, 10, kernel_size=3, stride=2, padding=1, output_padding=1),
        )
        self.conv_x2 = nn.Sequential(
            nn.Conv2d(1, 10, 5, stride=1, padding=2),
            nn.Conv2d(10, 10, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(10, 40, 3, stride=2, padding=1),
            nn.Conv2d(40, 20, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(20, 20, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(20, 10, kernel_size=3, stride=2, padding=1, output_padding=1),
        )
        self.h_h = nn.Sequential(
            nn.Conv2d(60, 30, 3, padding=1),
            nn.Conv2d(30, 20, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(20, 20, 3, padding=1),
        )
        self.res_part1 = res_part(60, 60)
        self.res_part2 = res_part(60, 60)

    def forward(self, xt1, meas=None, nor_meas=None, PhiTy=None, mask3d_batch=None, h=None, cs_rate=28):
        ht = h
        xt = xt1

        step = 2
        [bs, nC, row, col] = xt1.shape

        out = xt1
        x11 = self.conv_x1(torch.unsqueeze(nor_meas, 1))
        for i in range(cs_rate - 1):
            d1 = torch.zeros(bs, row, col).cuda()
            d2 = torch.zeros(bs, row, col).cuda()
            for ii in range(i + 1):
                d1 = d1 + torch.mul(mask3d_batch[:, ii, :, :], out[:, ii, :, :])
            for ii in range(i + 2, cs_rate):
                d2 = d2 + torch.mul(mask3d_batch[:, ii, :, :], torch.squeeze(nor_meas))
            x12 = self.conv_x2(torch.unsqueeze(meas - d1 - d2, 1))

            x2 = self.extract_feature1(xt)
            h = torch.cat([ht, x11, x12, x2], dim=1)
            h = self.res_part1(h)
            h = self.res_part2(h)
            ht = self.h_h(h)
            xt = self.up_feature1(h)
            out = torch.cat([out, xt], dim=1)

        return out, ht


class backrnn(nn.Module):

    def __init__(self):
        super(backrnn, self).__init__()
        self.extract_feature1 = down_feature(1, 20)
        self.up_feature1 = up_feature(60, 1)
        self.conv_x1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=1, padding=2),
            nn.Conv2d(16, 32, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 40, 3, stride=2, padding=1),
            nn.Conv2d(40, 20, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(20, 20, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(20, 10, kernel_size=3, stride=2, padding=1, output_padding=1),
        )
        self.conv_x2 = nn.Sequential(
            nn.Conv2d(1, 10, 5, stride=1, padding=2),
            nn.Conv2d(10, 10, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(10, 40, 3, stride=2, padding=1),
            nn.Conv2d(40, 20, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(20, 20, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(20, 10, kernel_size=3, stride=2, padding=1, output_padding=1),
        )
        self.h_h = nn.Sequential(
            nn.Conv2d(60, 30, 3, padding=1),
            nn.Conv2d(30, 20, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(20, 20, 3, padding=1),
        )
        self.res_part1 = res_part(60, 60)
        self.res_part2 = res_part(60, 60)

    def forward(self, xt8, meas=None, nor_meas=None, PhiTy=None, mask3d_batch=None, h=None, cs_rate=28):
        ht = h

        step = 2
        [bs, nC, row, col] = xt8.shape

        xt = torch.unsqueeze(xt8[:, cs_rate - 1, :, :], 1)

        out = torch.zeros(bs, cs_rate, row, col).cuda()
        out[:, cs_rate - 1, :, :] = xt[:, 0, :, :]
        x11 = self.conv_x1(torch.unsqueeze(nor_meas, 1))
        for i in range(cs_rate - 1):
            d1 = torch.zeros(bs, row, col).cuda()
            d2 = torch.zeros(bs, row, col).cuda()
            for ii in range(i + 1):
                d1 = d1 + torch.mul(mask3d_batch[:, cs_rate - 1 - ii, :, :], out[:, cs_rate - 1 - ii, :, :].clone())
            for ii in range(i + 2, cs_rate):
                d2 = d2 + torch.mul(mask3d_batch[:, cs_rate - 1 - ii, :, :], xt8[:, cs_rate - 1 - ii, :, :].clone())
            x12 = self.conv_x2(torch.unsqueeze(meas - d1 - d2, 1))

            x2 = self.extract_feature1(xt)
            h = torch.cat([ht, x11, x12, x2], dim=1)
            h = self.res_part1(h)
            h = self.res_part2(h)
            ht = self.h_h(h)
            xt = self.up_feature1(h)

            out[:, cs_rate - 2 - i, :, :] = xt[:, 0, :, :]

        return out

def shift_gt_back(inputs, step=2):  # input [bs,256,310]  output [bs, 28, 256, 256]
    [bs, nC, row, col] = inputs.shape
    output = torch.zeros(bs, nC, row, col - (nC - 1) * step).cuda().float()
    for i in range(nC):
        output[:, i, :, :] = inputs[:, i, :, step * i:step * i + col - (nC - 1) * step]
    return output

def shift(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    if inputs.is_cuda:
        output = torch.zeros(bs, nC, row, col + (nC - 1) * step).cuda().float()
    else:
        output = torch.zeros(bs, nC, row, col + (nC - 1) * step).float()
    for i in range(nC):
        output[:, i, :, step * i:step * i + col] = inputs[:, i, :, :]
    return output

class BIRNAT(nn.Module):

    def __init__(self):
        super(BIRNAT, self).__init__()
        self.cs_rate = 28
        self.first_frame_net = cnn1(self.cs_rate).cuda()
        self.rnn1 = forward_rnn().cuda()
        self.rnn2 = backrnn().cuda()

    def gen_meas_torch(self, meas, shift_mask):
        batch_size, H = meas.shape[0:2]
        mask_s = torch.sum(shift_mask, 1)
        nor_meas = torch.div(meas, mask_s)
        temp = torch.mul(torch.unsqueeze(nor_meas, dim=1).expand([batch_size, 28, H, shift_mask.shape[3]]), shift_mask)
        return nor_meas, temp

    def forward(self, meas, shift_mask=None):
        if shift_mask==None:
            shift_mask = torch.zeros(1, 28, 256, 310).cuda()
        H, W = meas.shape[-2:]
        nor_meas, PhiTy = self.gen_meas_torch(meas, shift_mask)
        h0 = torch.zeros(meas.shape[0], 20, H, W).cuda()
        xt1 = self.first_frame_net(meas, nor_meas, PhiTy)
        model_out1, h1 = self.rnn1(xt1, meas, nor_meas, PhiTy, shift_mask, h0, self.cs_rate)
        model_out2 = self.rnn2(model_out1, meas, nor_meas, PhiTy, shift_mask, h1, self.cs_rate)
        model_out2 = shift_gt_back(model_out2)

        return model_out2
