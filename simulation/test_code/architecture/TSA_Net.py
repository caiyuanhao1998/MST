import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
_NORM_BONE = False


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

_NORM_ATTN = True
_NORM_FC = False


class TSA_Transform(nn.Module):
    """ Spectral-Spatial Self-Attention """

    def __init__(self, uSpace, inChannel, outChannel, nHead, uAttn, mode=[0, 1], flag_mask=False, gamma_learn=False):
        super(TSA_Transform, self).__init__()
        ''' ------------------------------------------
        uSpace:
            uHeight: the [-2] dim of the 3D tensor
            uWidth: the [-1] dim of the 3D tensor
        inChannel: 
            the number of Channel of the input tensor
        outChannel: 
            the number of Channel of the output tensor
        nHead: 
            the number of Head of the input tensor
        uAttn:
            uSpatial: the dim of the spatial features
            uSpectral: the dim of the spectral features
        mask:
            The Spectral Smoothness Mask
        {mode} and {gamma_learn} is just for variable selection
        ------------------------------------------ '''

        self.nHead = nHead
        self.uAttn = uAttn
        self.outChannel = outChannel
        self.uSpatial = nn.Parameter(torch.tensor(float(uAttn[0])), requires_grad=False)
        self.uSpectral = nn.Parameter(torch.tensor(float(uAttn[1])), requires_grad=False)
        self.mask = nn.Parameter(Spectral_Mask(outChannel), requires_grad=False) if flag_mask else None
        self.attn_scale = nn.Parameter(torch.tensor(1.1), requires_grad=False) if flag_mask else None
        self.gamma = nn.Parameter(torch.tensor(1.0), requires_grad=gamma_learn)

        if sum(mode) > 0:
            down_sample = []
            scale = 1
            cur_channel = outChannel
            for i in range(sum(mode)):
                scale *= 2
                down_sample.append(conv_block(cur_channel, 2 * cur_channel, 3, 2, 1, _NORM_ATTN))
                cur_channel = 2 * cur_channel
            self.cur_channel = cur_channel
            self.down_sample = nn.Sequential(*down_sample)
            self.up_sample = nn.ConvTranspose2d(outChannel * scale, outChannel, scale, scale)
        else:
            self.down_sample = None
            self.up_sample = None

        spec_dim = int(uSpace[0] / 4 - 3) * int(uSpace[1] / 4 - 3)
        self.preproc = conv1x1_block(inChannel, outChannel, _NORM_ATTN)
        self.query_x = Feature_Spatial(outChannel, nHead, int(uSpace[1] / 4), uAttn[0], mode)
        self.query_y = Feature_Spatial(outChannel, nHead, int(uSpace[0] / 4), uAttn[0], mode)
        self.query_lambda = Feature_Spectral(outChannel, nHead, spec_dim, uAttn[1])
        self.key_x = Feature_Spatial(outChannel, nHead, int(uSpace[1] / 4), uAttn[0], mode)
        self.key_y = Feature_Spatial(outChannel, nHead, int(uSpace[0] / 4), uAttn[0], mode)
        self.key_lambda = Feature_Spectral(outChannel, nHead, spec_dim, uAttn[1])
        self.value = conv1x1_block(outChannel, nHead * outChannel, _NORM_ATTN)
        self.aggregation = nn.Linear(nHead * outChannel, outChannel)

    def forward(self, image):
        feat = self.preproc(image)
        feat_qx = self.query_x(feat, 'X')
        feat_qy = self.query_y(feat, 'Y')
        feat_qlambda = self.query_lambda(feat)
        feat_kx = self.key_x(feat, 'X')
        feat_ky = self.key_y(feat, 'Y')
        feat_klambda = self.key_lambda(feat)
        feat_value = self.value(feat)

        feat_qx = torch.cat(torch.split(feat_qx, 1, dim=1)).squeeze(dim=1)
        feat_qy = torch.cat(torch.split(feat_qy, 1, dim=1)).squeeze(dim=1)
        feat_kx = torch.cat(torch.split(feat_kx, 1, dim=1)).squeeze(dim=1)
        feat_ky = torch.cat(torch.split(feat_ky, 1, dim=1)).squeeze(dim=1)
        feat_qlambda = torch.cat(torch.split(feat_qlambda, self.uAttn[1], dim=-1))
        feat_klambda = torch.cat(torch.split(feat_klambda, self.uAttn[1], dim=-1))
        feat_value = torch.cat(torch.split(feat_value, self.outChannel, dim=1))

        energy_x = torch.bmm(feat_qx, feat_kx.permute(0, 2, 1)) / torch.sqrt(self.uSpatial)
        energy_y = torch.bmm(feat_qy, feat_ky.permute(0, 2, 1)) / torch.sqrt(self.uSpatial)
        energy_lambda = torch.bmm(feat_qlambda, feat_klambda.permute(0, 2, 1)) / torch.sqrt(self.uSpectral)

        attn_x = F.softmax(energy_x, dim=-1)
        attn_y = F.softmax(energy_y, dim=-1)
        attn_lambda = F.softmax(energy_lambda, dim=-1)
        if self.mask is not None:
            attn_lambda = (attn_lambda + self.mask) / torch.sqrt(self.attn_scale)

        pro_feat = feat_value if self.down_sample is None else self.down_sample(feat_value)
        batchhead, dim_c, dim_x, dim_y = pro_feat.size()
        attn_x_repeat = attn_x.unsqueeze(dim=1).repeat(1, dim_c, 1, 1).view(-1, dim_x, dim_x)
        attn_y_repeat = attn_y.unsqueeze(dim=1).repeat(1, dim_c, 1, 1).view(-1, dim_y, dim_y)
        pro_feat = pro_feat.view(-1, dim_x, dim_y)
        pro_feat = torch.bmm(pro_feat, attn_y_repeat.permute(0, 2, 1))
        pro_feat = torch.bmm(pro_feat.permute(0, 2, 1), attn_x_repeat.permute(0, 2, 1)).permute(0, 2, 1)
        pro_feat = pro_feat.view(batchhead, dim_c, dim_x, dim_y)

        if self.up_sample is not None:
            pro_feat = self.up_sample(pro_feat)
        _, _, dim_x, dim_y = pro_feat.size()
        pro_feat = pro_feat.contiguous().view(batchhead, self.outChannel, -1).permute(0, 2, 1)
        pro_feat = torch.bmm(pro_feat, attn_lambda.permute(0, 2, 1)).permute(0, 2, 1)
        pro_feat = pro_feat.view(batchhead, self.outChannel, dim_x, dim_y)
        pro_feat = torch.cat(torch.split(pro_feat, int(batchhead / self.nHead), dim=0), dim=1).permute(0, 2, 3, 1)
        pro_feat = self.aggregation(pro_feat).permute(0, 3, 1, 2)
        out = self.gamma * pro_feat + feat
        return out, (attn_x, attn_y, attn_lambda)


class Feature_Spatial(nn.Module):
    """ Spatial Feature Generation Component """

    def __init__(self, inChannel, nHead, shiftDim, outDim, mode):
        super(Feature_Spatial, self).__init__()
        kernel = [(1, 5), (3, 5)]
        stride = [(1, 2), (2, 2)]
        padding = [(0, 2), (1, 2)]
        self.conv1 = conv_block(inChannel, nHead, kernel[mode[0]], stride[mode[0]], padding[mode[0]], _NORM_ATTN)
        self.conv2 = conv_block(nHead, nHead, kernel[mode[1]], stride[mode[1]], padding[mode[1]], _NORM_ATTN)
        self.fully = fully_block(shiftDim, outDim, _NORM_FC)

    def forward(self, image, direction):
        if direction == 'Y':
            image = image.permute(0, 1, 3, 2)
        feat = self.conv1(image)
        feat = self.conv2(feat)
        feat = self.fully(feat)
        return feat


class Feature_Spectral(nn.Module):
    """ Spectral Feature Generation Component """

    def __init__(self, inChannel, nHead, viewDim, outDim):
        super(Feature_Spectral, self).__init__()
        self.inChannel = inChannel
        self.conv1 = conv_block(inChannel, inChannel, 5, 2, 0, _NORM_ATTN)
        self.conv2 = conv_block(inChannel, inChannel, 5, 2, 0, _NORM_ATTN)
        self.fully = fully_block(viewDim, int(nHead * outDim), _NORM_FC)

    def forward(self, image):
        bs = image.size(0)
        feat = self.conv1(image)
        feat = self.conv2(feat)
        feat = feat.view(bs, self.inChannel, -1)
        feat = self.fully(feat)
        return feat


def Spectral_Mask(dim_lambda):
    '''After put the available data into the model, we use this mask to avoid outputting the estimation of itself.'''
    orig = (np.cos(np.linspace(-1, 1, num=2 * dim_lambda - 1) * np.pi) + 1.0) / 2.0
    att = np.zeros((dim_lambda, dim_lambda))
    for i in range(dim_lambda):
        att[i, :] = orig[dim_lambda - 1 - i:2 * dim_lambda - 1 - i]
    AM_Mask = torch.from_numpy(att.astype(np.float32)).unsqueeze(0)
    return AM_Mask

class TSA_Net(nn.Module):

    def __init__(self, in_ch=28, out_ch=28):
        super(TSA_Net, self).__init__()

        self.tconv_down1 = Encoder_Triblock(in_ch, 64, False)
        self.tconv_down2 = Encoder_Triblock(64, 128, False)
        self.tconv_down3 = Encoder_Triblock(128, 256)
        self.tconv_down4 = Encoder_Triblock(256, 512)

        self.bottom1 = conv_block(512, 1024)
        self.bottom2 = conv_block(1024, 1024)

        self.tconv_up4 = Decoder_Triblock(1024, 512)
        self.tconv_up3 = Decoder_Triblock(512, 256)
        self.transform3 = TSA_Transform((64, 64), 256, 256, 8, (64, 80), [0, 0])
        self.tconv_up2 = Decoder_Triblock(256, 128)
        self.transform2 = TSA_Transform((128, 128), 128, 128, 8, (64, 40), [1, 0])
        self.tconv_up1 = Decoder_Triblock(128, 64)
        self.transform1 = TSA_Transform((256, 256), 64, 28, 8, (48, 30), [1, 1], True)

        self.conv_last = nn.Conv2d(out_ch, out_ch, 1)
        self.afn_last = nn.Sigmoid()

    def forward(self, x, input_mask=None):
        enc1, enc1_pre = self.tconv_down1(x)
        enc2, enc2_pre = self.tconv_down2(enc1)
        enc3, enc3_pre = self.tconv_down3(enc2)
        enc4, enc4_pre = self.tconv_down4(enc3)
        # enc5,enc5_pre = self.tconv_down5(enc4)

        bottom = self.bottom1(enc4)
        bottom = self.bottom2(bottom)

        # dec5 = self.tconv_up5(bottom,enc5_pre)
        dec4 = self.tconv_up4(bottom, enc4_pre)
        dec3 = self.tconv_up3(dec4, enc3_pre)
        dec3, _ = self.transform3(dec3)
        dec2 = self.tconv_up2(dec3, enc2_pre)
        dec2, _ = self.transform2(dec2)
        dec1 = self.tconv_up1(dec2, enc1_pre)
        dec1, _ = self.transform1(dec1)

        dec1 = self.conv_last(dec1)
        output = self.afn_last(dec1)

        return output


class Encoder_Triblock(nn.Module):
    def __init__(self, inChannel, outChannel, flag_res=True, nKernal=3, nPool=2, flag_Pool=True):
        super(Encoder_Triblock, self).__init__()

        self.layer1 = conv_block(inChannel, outChannel, nKernal, flag_norm=_NORM_BONE)
        if flag_res:
            self.layer2 = Res2Net(outChannel, int(outChannel / 4))
        else:
            self.layer2 = conv_block(outChannel, outChannel, nKernal, flag_norm=_NORM_BONE)

        self.pool = nn.MaxPool2d(nPool) if flag_Pool else None

    def forward(self, x):
        feat = self.layer1(x)
        feat = self.layer2(feat)

        feat_pool = self.pool(feat) if self.pool is not None else feat
        return feat_pool, feat


class Decoder_Triblock(nn.Module):
    def __init__(self, inChannel, outChannel, flag_res=True, nKernal=3, nPool=2, flag_Pool=True):
        super(Decoder_Triblock, self).__init__()

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(inChannel, outChannel, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        if flag_res:
            self.layer2 = Res2Net(int(outChannel * 2), int(outChannel / 2))
        else:
            self.layer2 = conv_block(outChannel * 2, outChannel * 2, nKernal, flag_norm=_NORM_BONE)
        self.layer3 = conv_block(outChannel * 2, outChannel, nKernal, flag_norm=_NORM_BONE)

    def forward(self, feat_dec, feat_enc):
        feat_dec = self.layer1(feat_dec)
        diffY = feat_enc.size()[2] - feat_dec.size()[2]
        diffX = feat_enc.size()[3] - feat_dec.size()[3]
        if diffY != 0 or diffX != 0:
            print('Padding for size mismatch ( Enc:', feat_enc.size(), 'Dec:', feat_dec.size(), ')')
            feat_dec = F.pad(feat_dec, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        feat = torch.cat([feat_dec, feat_enc], dim=1)
        feat = self.layer2(feat)
        feat = self.layer3(feat)
        return feat