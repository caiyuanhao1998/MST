import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import einsum
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
from collections import defaultdict, Counter
import numpy as np
from tqdm import tqdm
import random

def uniform(a, b, shape, device='cuda'):
    return (b - a) * torch.rand(shape, device=device) + a

class AsymmetricTransform:

    def Q(self, *args, **kwargs):
        raise NotImplementedError('Query transform not implemented')

    def K(self, *args, **kwargs):
        raise NotImplementedError('Key transform not implemented')

class LSH:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError('LSH scheme not implemented')

    def compute_hash_agreement(self, q_hash, k_hash):
        return (q_hash == k_hash).min(dim=-1)[0].sum(dim=-1)

class XBOXPLUS(AsymmetricTransform):

    def set_norms(self, x):
        self.x_norms = x.norm(p=2, dim=-1, keepdim=True)
        self.MX = torch.amax(self.x_norms, dim=-2, keepdim=True)

    def X(self, x):
        device = x.device
        ext = torch.sqrt((self.MX**2).to(device) - (self.x_norms**2).to(device))
        zero = torch.tensor(0.0, device=x.device).repeat(x.shape[:-1], 1).unsqueeze(-1)
        return torch.cat((x, ext, zero), -1)

def lsh_clustering(x, n_rounds, r=1):
    salsh = SALSH(n_rounds=n_rounds, dim=x.shape[-1], r=r, device=x.device)
    x_hashed = salsh(x).reshape((n_rounds,) + x.shape[:-1])
    return x_hashed.argsort(dim=-1)

class SALSH(LSH):
    def __init__(self, n_rounds, dim, r, device='cuda'):
        super(SALSH, self).__init__()
        self.alpha = torch.normal(0, 1, (dim, n_rounds), device=device)
        self.beta = uniform(0, r, shape=(1, n_rounds), device=device)
        self.dim = dim
        self.r = r

    def __call__(self, vecs):
        projection = vecs @ self.alpha
        projection_shift = projection + self.beta
        projection_rescale = projection_shift / self.r
        return projection_rescale.permute(2, 0, 1)

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2

    variance = scale / denom

    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


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

def batch_scatter(output, src, dim, index):
    """
    :param output: [b,n,c]
    :param src: [b,n,c]
    :param dim: int
    :param index: [b,n]
    :return: output: [b,n,c]
    """
    b,k,c = src.shape
    index = index[:, :, None].expand(-1, -1, c)
    output, src, index = map(lambda t: rearrange(t, 'b k c -> (b c) k'), (output, src, index))
    output.scatter_(dim,index,src)
    output = rearrange(output, '(b c) k -> b k c', b=b)
    return output

def batch_gather(x, index, dim):
    """
    :param x: [b,n,c]
    :param index: [b,n//2]
    :param dim: int
    :return: output: [b,n//2,c]
    """
    b,n,c = x.shape
    index = index[:,:,None].expand(-1,-1,c)
    x, index = map(lambda t: rearrange(t, 'b n c -> (b c) n'), (x, index))
    output = torch.gather(x,dim,index)
    output = rearrange(output, '(b c) n -> b n c', b=b)
    return output

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)

class SAH_MSA(nn.Module):
    def __init__(self, heads=4, n_rounds=2, channels=64, patch_size=144,
                r=1):
        super(SAH_MSA, self).__init__()
        self.heads = heads
        self.n_rounds = n_rounds
        inner_dim = channels*3
        self.to_q = nn.Linear(channels, inner_dim, bias=False)
        self.to_k = nn.Linear(channels, inner_dim, bias=False)
        self.to_v = nn.Linear(channels, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, channels, bias=False)

        self.xbox_plus = XBOXPLUS()
        self.clustering_params = {
            'r': r,
            'n_rounds': self.n_rounds
        }
        self.q_attn_size = patch_size[0] * patch_size[1]
        self.k_attn_size = patch_size[0] * patch_size[1]

    def forward(self, input):
        """
        :param input: [b,n,c]
        :return: output: [b,n,c]
        """

        B, N, C_inp = input.shape
        query = self.to_q(input)
        key = self.to_k(input)
        value = self.to_v(input)
        input_hash = input.view(B, N, self.heads, C_inp//self.heads)
        x_hash = rearrange(input_hash, 'b t h e -> (b h) t e')
        bs, x_seqlen, dim = x_hash.shape
        with torch.no_grad():
            self.xbox_plus.set_norms(x_hash)
            Xs = self.xbox_plus.X(x_hash)
            x_positions = lsh_clustering(Xs, **self.clustering_params)
            x_positions = x_positions.reshape(self.n_rounds, bs, -1)

        del Xs

        C = query.shape[-1]
        query = query.view(B, N, self.heads, C // self.heads)
        key = key.view(B, N, self.heads, C // self.heads)
        value = value.view(B, N, self.heads, C // self.heads)

        query = rearrange(query, 'b t h e -> (b h) t e')   # [bs, q_seqlen,c]
        key = rearrange(key, 'b t h e -> (b h) t e')
        value = rearrange(value, 'b s h d -> (b h) s d')

        bs, q_seqlen, dim = query.shape
        bs, k_seqlen, dim = key.shape
        v_dim = value.shape[-1]

        x_rev_positions = torch.argsort(x_positions, dim=-1)
        x_offset = torch.arange(bs, device=query.device).unsqueeze(-1) * x_seqlen
        x_flat = (x_positions + x_offset).reshape(-1)

        s_queries = query.reshape(-1, dim).index_select(0, x_flat).reshape(-1, self.q_attn_size, dim)
        s_keys = key.reshape(-1, dim).index_select(0, x_flat).reshape(-1, self.k_attn_size, dim)
        s_values = value.reshape(-1, v_dim).index_select(0, x_flat).reshape(-1, self.k_attn_size, v_dim)

        inner = s_queries @ s_keys.transpose(2, 1)
        norm_factor = 1
        inner = inner / norm_factor

        # free memory
        del x_positions

        # softmax denominator
        dots_logsumexp = torch.logsumexp(inner, dim=-1, keepdim=True)
        # softmax
        dots = torch.exp(inner - dots_logsumexp)
        # dropout

        # n_rounds outs
        bo = (dots @ s_values).reshape(self.n_rounds, bs, q_seqlen, -1)

        # undo sort
        x_offset = torch.arange(bs * self.n_rounds, device=query.device).unsqueeze(-1) * x_seqlen
        x_rev_flat = (x_rev_positions.reshape(-1, x_seqlen) + x_offset).reshape(-1)
        o = bo.reshape(-1, v_dim).index_select(0, x_rev_flat).reshape(self.n_rounds, bs, q_seqlen, -1)

        slogits = dots_logsumexp.reshape(self.n_rounds, bs, -1)
        logits = torch.gather(slogits, 2, x_rev_positions)

        # free memory
        del x_rev_positions

        # weighted sum multi-round attention
        probs = torch.exp(logits - torch.logsumexp(logits, dim=0, keepdim=True))
        out = torch.sum(o * probs.unsqueeze(-1), dim=0)
        out = rearrange(out, '(b h) t d -> b t h d', h=self.heads)
        out = out.reshape(B, N, -1)
        out = self.to_out(out)

        return out

class SAHAB(nn.Module):
    def __init__(
            self,
            dim,
            patch_size=(16, 16),
            heads=8,
            shift_size=0,
            sparse=False
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        self.attn = PreNorm(dim, SAH_MSA(heads=heads, n_rounds=2, r=1, channels=dim, patch_size=patch_size))
        self.ffn = PreNorm(dim, FeedForward(dim=dim))
        self.shift_size = shift_size
        self.patch_size = patch_size
        self.sparse = sparse

    def forward(self, x, mask=None):
        """
        x: [b,h,w,c]
        mask: [b,h,w]
        return out: [b,h,w,c]
        """
        b,h,w,c = x.shape
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            mask = torch.roll(mask, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        w_size = self.patch_size

        # Split into large patches
        x = rearrange(x, 'b (nh hh) (nw ww) c-> b (nh nw) (hh ww c)', hh=w_size[0] * 2, ww=w_size[1] * 2)
        mask = rearrange(mask, 'b (nh hh) (nw ww) -> b (nh nw) (hh ww)', hh=w_size[0] * 2, ww=w_size[1] * 2)
        N = x.shape[1]

        mask = torch.mean(mask,dim=2,keepdim=False) # [b,nh*nw]
        if self.sparse:
            mask_select = mask.topk(mask.shape[1] // 2, dim=1)[1]  # [b,nh*nw//2]
            x_select = batch_gather(x, mask_select, 1)  # [b,nh*nw//2,hh*ww*c]
            x_select = x_select.reshape(b*N//2,-1,c)
            x_select = self.attn(x_select)+x_select
            x_select = x_select.view(b,N//2,-1)
            x = batch_scatter(x.clone(), x_select, 1, mask_select)
        else:
            x = x.view(b*N,-1,c)
            x = self.attn(x) + x
            x = x.view(b, N, -1)
        x = rearrange(x, 'b (nh nw) (hh ww c) -> b (nh hh) (nw ww) c', nh=h//(w_size[0] * 2), hh=w_size[0] * 2, ww=w_size[1] * 2)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = self.ffn(x) + x

        return x


class SAHABs(nn.Module):
    def __init__(
            self,
            dim,
            patch_size=(8, 8),
            heads=8,
            num_blocks=2,
            sparse=False
    ):
        super().__init__()
        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                SAHAB(heads=heads, dim=dim, patch_size=patch_size,sparse=sparse,
                                     shift_size=0 if (_ % 2 == 0) else patch_size[0]))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x, mask=None):
        """
        x: [b,c,h,w]
        mask: [b,1,h,w]
        return x: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        mask = mask.squeeze(1)
        for block in self.blocks:
            x = block(x, mask)
        x = x.permute(0, 3, 1, 2)
        return x

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels):
        super(ASPP, self).__init__()
        modules = []

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

class Sparsity_Estimator(nn.Module):
    def __init__(self, dim=28, expand=2, sparse=False):
        super(Sparsity_Estimator, self).__init__()
        self.dim = dim
        self.stage = 2
        self.sparse = sparse
        
        # Input projection
        self.in_proj = nn.Conv2d(28, dim, 1, 1, 0, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(2):
            self.encoder_layers.append(nn.ModuleList([
                nn.Conv2d(dim_stage, dim_stage * expand, 1, 1, 0, bias=False),
                nn.Conv2d(dim_stage * expand, dim_stage * expand, 3, 2, 1, bias=False, groups=dim_stage * expand),
                nn.Conv2d(dim_stage * expand, dim_stage*expand, 1, 1, 0, bias=False),
            ]))
            dim_stage *= 2

        # Bottleneck
        self.bottleneck = ASPP(dim_stage, [3,6], dim_stage)

        # Decoder:
        self.decoder_layers = nn.ModuleList([])
        for i in range(2):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_stage // 2, dim_stage, 1, 1, 0, bias=False),
                nn.Conv2d(dim_stage, dim_stage, 3, 1, 1, bias=False, groups=dim_stage),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, 0, bias=False),
            ]))
            dim_stage //= 2

        # Output projection
        if sparse:
            self.out_conv2 = nn.Conv2d(self.dim, self.dim+1, 3, 1, 1, bias=False)
        else:
            self.out_conv2 = nn.Conv2d(self.dim, self.dim, 3, 1, 1, bias=False)
            
        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        # Input projection
        fea = self.lrelu(self.in_proj(x))
        # Encoder
        fea_encoder = []  # [c 2c 4c 8c]
        for (Conv1, Conv2, Conv3) in self.encoder_layers:
            fea_encoder.append(fea)
            fea = Conv3(self.lrelu(Conv2(self.lrelu(Conv1(fea)))))
        # Bottleneck
        fea = self.bottleneck(fea)+fea
        # Decoder
        for i, (FeaUpSample, Conv1, Conv2, Conv3) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Conv3(self.lrelu(Conv2(self.lrelu(Conv1(fea)))))
            fea = fea + fea_encoder[self.stage-1-i]
        # Output projection
        out = self.out_conv2(fea)
        if self.sparse:
            error_map = out[:,-1:,:,:]
            return out[:,:-1], error_map
        return out

class CST(nn.Module):
    def __init__(self, dim=28, stage=2, num_blocks=[2, 2, 2], sparse=False):
        super(CST, self).__init__()
        self.dim = dim
        self.stage = stage
        self.sparse = sparse

        # Fution physical mask and shifted measurement
        self.fution = nn.Conv2d(56, 28, 1, 1, 0, bias=False)

        # Sparsity Estimator
        if num_blocks==[2,4,6]:
            self.fe = nn.Sequential(Sparsity_Estimator(dim=28,expand=2,sparse=False),
                        Sparsity_Estimator(dim=28, expand=2, sparse=sparse))
        else:
            self.fe = Sparsity_Estimator(dim=28, expand=2, sparse=sparse)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                SAHABs(dim=dim_stage, num_blocks=num_blocks[i], heads=dim_stage // dim, sparse=sparse),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
                nn.AvgPool2d(kernel_size=2, stride=2),
            ]))
            dim_stage *= 2

        # Bottleneck
        self.bottleneck = SAHABs(
            dim=dim_stage, heads=dim_stage // dim, num_blocks=num_blocks[-1], sparse=sparse)

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(stage):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                SAHABs(dim=dim_stage // 2, num_blocks=num_blocks[stage - 1 - i],
                      heads=(dim_stage // 2) // dim, sparse=sparse),
            ]))
            dim_stage //= 2

        # Output projection
        self.out_proj = nn.Conv2d(self.dim, dim, 3, 1, 1, bias=False)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, mask=None):
        """
        x: [b,c,h,w]
        mask: [b,c,h,w]
        return out:[b,c,h,w]
        """
        b, c, h, w = x.shape

        # Fution
        x = self.fution(torch.cat([x,mask],dim=1))

        # Feature Extraction
        if self.sparse:
            fea,mask = self.fe(x)
        else:
            fea = self.fe(x)
            mask = torch.randn((b,1,h,w)).cuda()

        # Encoder
        fea_encoder = []
        masks = []
        for (Blcok, FeaDownSample, MaskDownSample) in self.encoder_layers:
            fea = Blcok(fea, mask)
            masks.append(mask)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)
            mask = MaskDownSample(mask)

        # Bottleneck
        fea = self.bottleneck(fea, mask)

        # Decoder
        for i, (FeaUpSample, Blcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = fea + fea_encoder[self.stage - 1 - i]
            mask = masks[self.stage - 1 - i]
            fea = Blcok(fea, mask)

        # Output projection
        out = self.out_proj(fea) + x
        return out












