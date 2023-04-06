# credits: https://github.com/SHI-Labs/StyleNAT

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()
        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)
        return out


class EqualConv2d(nn.Module):
    def __init__(
            self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )


class EqualLinear(nn.Module):
    def __init__(
            self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)
        return out


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * math.sqrt(2)


class ModulatedConv2d(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            demodulate=True,
            upsample=False,
            downsample=False,
            blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(
                pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample}, downsample={self.downsample})'
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(
                input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.reshape(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal Positional Embedding 1D or 2D (SPE/SPE2d).
    This module is a modified from:
    https://github.com/pytorch/fairseq/blob/master/fairseq/modules/sinusoidal_positional_embedding.py # noqa
    Based on the original SPE in single dimension, we implement a 2D sinusoidal
    positional encodding (SPE2d), as introduced in Positional Encoding as
    Spatial Inductive Bias in GANs, CVPR'2021.
    Args:
        embedding_dim (int): The number of dimensions for the positional
            encoding.
        padding_idx (int | list[int]): The index for the padding contents. The
            padding positions will obtain an encoding vector filling in zeros.
        init_size (int, optional): The initial size of the positional buffer.
            Defaults to 1024.
        div_half_dim (bool, optional): If true, the embedding will be divided
            by :math:`d/2`. Otherwise, it will be divided by
            :math:`(d/2 -1)`. Defaults to False.
        center_shift (int | None, optional): Shift the center point to some
            index. Defaults to None.
    """

    def __init__(self,
                 embedding_dim,
                 padding_idx,
                 init_size=1024,
                 div_half_dim=False,
                 center_shift=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.div_half_dim = div_half_dim
        self.center_shift = center_shift

        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size, embedding_dim, padding_idx, self.div_half_dim)

        self.register_buffer('_float_tensor', torch.FloatTensor(1))

        self.max_positions = int(1e5)

    @staticmethod
    def get_embedding(num_embeddings,
                      embedding_dim,
                      padding_idx=None,
                      div_half_dim=False):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        assert embedding_dim % 2 == 0, (
            'In this version, we request '
            f'embedding_dim divisible by 2 but got {embedding_dim}')

        # there is a little difference from the original paper.
        half_dim = embedding_dim // 2
        if not div_half_dim:
            emb = np.log(10000) / (half_dim - 1)
        else:
            emb = np.log(1e4) / half_dim
        # compute exp(-log10000 / d * i)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(
            num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)],
                        dim=1).view(num_embeddings, -1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0

        return emb

    def forward(self, input, **kwargs):
        """Input is expected to be of size [bsz x seqlen].
        Returned tensor is expected to be of size  [bsz x seq_len x emb_dim]
        """
        assert input.dim() == 2 or input.dim(
        ) == 4, 'Input dimension should be 2 (1D) or 4(2D)'

        if input.dim() == 4:
            return self.make_grid2d_like(input, **kwargs)

        b, seq_len = input.shape
        max_pos = self.padding_idx + 1 + seq_len

        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embedding if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos, self.embedding_dim, self.padding_idx)
        self.weights = self.weights.to(self._float_tensor)

        positions = self.make_positions(input, self.padding_idx).to(
            self._float_tensor.device)

        return self.weights.index_select(0, positions.view(-1)).view(
            b, seq_len, self.embedding_dim).detach()

    def make_positions(self, input, padding_idx):
        mask = input.ne(padding_idx).int()
        return (torch.cumsum(mask, dim=1).type_as(mask) *
                mask).long() + padding_idx

    def make_grid2d(self, height, width, num_batches=1, center_shift=None):
        h, w = height, width
        # if `center_shift` is not given from the outside, use
        # `self.center_shift`
        if center_shift is None:
            center_shift = self.center_shift

        h_shift = 0
        w_shift = 0
        # center shift to the input grid
        if center_shift is not None:
            # if h/w is even, the left center should be aligned with
            # center shift
            if h % 2 == 0:
                h_left_center = h // 2
                h_shift = center_shift - h_left_center
            else:
                h_center = h // 2 + 1
                h_shift = center_shift - h_center

            if w % 2 == 0:
                w_left_center = w // 2
                w_shift = center_shift - w_left_center
            else:
                w_center = w // 2 + 1
                w_shift = center_shift - w_center

        # Note that the index is started from 1 since zero will be padding idx.
        # axis -- (b, h or w)
        x_axis = torch.arange(1, w + 1).unsqueeze(0).repeat(num_batches,
                                                            1) + w_shift
        y_axis = torch.arange(1, h + 1).unsqueeze(0).repeat(num_batches,
                                                            1) + h_shift

        # emb -- (b, emb_dim, h or w)
        x_emb = self(x_axis).transpose(1, 2)
        y_emb = self(y_axis).transpose(1, 2)

        # make grid for x/y axis
        # Note that repeat will copy data. If use learned emb, expand may be
        # better.
        x_grid = x_emb.unsqueeze(2).repeat(1, 1, h, 1)
        y_grid = y_emb.unsqueeze(3).repeat(1, 1, 1, w)

        # cat grid -- (b, 2 x emb_dim, h, w)
        grid = torch.cat([x_grid, y_grid], dim=1)
        return grid.detach()

    def make_grid2d_like(self, x, center_shift=None):
        """Input tensor with shape of (b, ..., h, w) Return tensor with shape
        of (b, 2 x emb_dim, h, w)
        Note that the positional embedding highly depends on the the function,
        ``make_positions``.
        """
        h, w = x.shape[-2:]
        grid = self.make_grid2d(h, w, x.size(0), center_shift)

        return grid.to(x)


class ConvLayer(nn.Sequential):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            downsample=False,
            blur_kernel=[1, 3, 3, 1],
            bias=True,
            activate=True,
            sn=False
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        if sn:
            # Not use equal conv2d when apply SN
            layers.append(
                spectral_norm(nn.Conv2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=self.padding,
                    stride=stride,
                    bias=bias and not activate,
                ))
            )
        else:
            layers.append(
                EqualConv2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=self.padding,
                    stride=stride,
                    bias=bias and not activate,
                )
            )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))
            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1], sn=False):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3, sn=sn)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True, sn=sn)

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        return out


def get_haar_wavelet(in_channels):
    haar_wav_l = 1 / (2 ** 0.5) * torch.ones(1, 2)
    haar_wav_h = 1 / (2 ** 0.5) * torch.ones(1, 2)
    haar_wav_h[0, 0] = -1 * haar_wav_h[0, 0]

    haar_wav_ll = haar_wav_l.T * haar_wav_l
    haar_wav_lh = haar_wav_h.T * haar_wav_l
    haar_wav_hl = haar_wav_l.T * haar_wav_h
    haar_wav_hh = haar_wav_h.T * haar_wav_h

    return haar_wav_ll, haar_wav_lh, haar_wav_hl, haar_wav_hh


class HaarTransform(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        ll, lh, hl, hh = get_haar_wavelet(in_channels)

        self.register_buffer('ll', ll)
        self.register_buffer('lh', lh)
        self.register_buffer('hl', hl)
        self.register_buffer('hh', hh)

    def forward(self, input):
        ll = upfirdn2d(input, self.ll, down=2)
        lh = upfirdn2d(input, self.lh, down=2)
        hl = upfirdn2d(input, self.hl, down=2)
        hh = upfirdn2d(input, self.hh, down=2)

        return torch.cat((ll, lh, hl, hh), 1)


class InverseHaarTransform(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        ll, lh, hl, hh = get_haar_wavelet(in_channels)

        self.register_buffer('ll', ll)
        self.register_buffer('lh', -lh)
        self.register_buffer('hl', -hl)
        self.register_buffer('hh', hh)

    def forward(self, input):
        ll, lh, hl, hh = input.chunk(4, 1)
        ll = upfirdn2d(ll, self.ll, up=2, pad=(1, 0, 1, 0))
        lh = upfirdn2d(lh, self.lh, up=2, pad=(1, 0, 1, 0))
        hl = upfirdn2d(hl, self.hl, up=2, pad=(1, 0, 1, 0))
        hh = upfirdn2d(hh, self.hh, up=2, pad=(1, 0, 1, 0))

        return ll + lh + hl + hh


class FromRGB(nn.Module):
    def __init__(self, out_channel, downsample=True, blur_kernel=[1, 3, 3, 1], sn=False):
        super().__init__()

        self.downsample = downsample

        if downsample:
            self.iwt = InverseHaarTransform(3)
            self.downsample = Downsample(blur_kernel)
            self.dwt = HaarTransform(3)

        self.conv = ConvLayer(3 * 4, out_channel, 1, sn=sn)

    def forward(self, input, skip=None):
        if self.downsample:
            input = self.iwt(input)
            input = self.downsample(input)
            input = self.dwt(input)

        out = self.conv(input)

        if skip is not None:
            out = out + skip

        return input, out


class Discriminator(nn.Module):
    def __init__(self,
                 size,
                 scale=1
                 ):
        super().__init__()
        channel_multiplier = 2
        sn = True
        blur_kernel = [1, 3, 3, 1]  # todo: blurring problem?

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }
        channels = {k: int(v * scale) for k, v in channels.items()}
        self.size = size

        self.dwt = HaarTransform(3)

        self.from_rgbs = nn.ModuleList()
        self.convs = nn.ModuleList()

        log_size = int(math.log(self.size, 2)) - 1

        in_channel = channels[self.size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            self.from_rgbs.append(FromRGB(in_channel, downsample=i != log_size, sn=sn))
            self.convs.append(ConvBlock(in_channel, out_channel, blur_kernel, sn=sn))

            in_channel = out_channel

        self.from_rgbs.append(FromRGB(channels[4], sn=sn))

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3, sn=sn)
        if sn:
            self.final_linear = nn.Sequential(
                spectral_norm(nn.Linear(channels[4] * 4 * 4, channels[4])),
                FusedLeakyReLU(channels[4]),
                spectral_norm(nn.Linear(channels[4], 1)),
            )
        else:
            self.final_linear = nn.Sequential(
                EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
                EqualLinear(channels[4], 1),
            )

    def forward(self, input):
        input = self.dwt(input)
        out = None

        for from_rgb, conv in zip(self.from_rgbs, self.convs):
            input, out = from_rgb(input, out)
            out = conv(out)

        _, out = self.from_rgbs[-1](input, out)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out
