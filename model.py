# -*- coding: utf-8 -*-
"""
Created on Tue Dec  11:00:00 2023

@author: chun
"""

import torch
import torch.nn as nn
import channel

""" def _image_normalization(norm_type):
    def _inner(tensor: torch.Tensor):
        if norm_type == 'nomalization':
            return tensor / 255.0
        elif norm_type == 'denormalization':
            return (tensor * 255.0).type(torch.FloatTensor)
        else:
            raise Exception('Unknown type of normalization')
    return _inner """


def null_print(x, x2=...):
    pass


_if_print = null_print


def ratio2filtersize(x: torch.Tensor, ratio):
    if x.dim() == 4:
        # before_size = np.prod(x.size()[1:])
        before_size = torch.prod(torch.tensor(x.size()[1:]))
    elif x.dim() == 3:
        # before_size = np.prod(x.size())
        before_size = torch.prod(torch.tensor(x.size()))
    else:
        raise Exception('Unknown size of input')
    encoder_temp = _Encoder(is_temp=True)
    z_temp = encoder_temp(x)
    # c = before_size * ratio / np.prod(z_temp.size()[-2:])
    # c = before_size * ratio / torch.prod(torch.tensor(z_temp.size()[-2:]))
    c = before_size * ratio / 2
    return int(c) * 2


class _ConvWithPReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(_ConvWithPReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.prelu = nn.PReLU()

        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.conv(x)
        x = self.prelu(x)
        return x


class _TransConvWithPReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activate=nn.PReLU(), padding=0,
                 output_padding=0):
        super(_TransConvWithPReLU, self).__init__()
        self.transconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.activate = activate
        if activate == nn.PReLU():
            nn.init.kaiming_normal_(self.transconv.weight, mode='fan_out', nonlinearity='leaky_relu')
        else:
            nn.init.xavier_normal_(self.transconv.weight)

    def forward(self, x):
        x = self.transconv(x)
        x = self.activate(x)
        return x


class _Encoder(nn.Module):
    def __init__(self, c=1, is_temp=False, P=1):
        super(_Encoder, self).__init__()
        self.is_temp = is_temp
        # self.imgae_normalization = _image_normalization(norm_type='nomalization')
        self.conv1 = _ConvWithPReLU(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.conv2 = _ConvWithPReLU(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv3 = _ConvWithPReLU(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.conv4 = _ConvWithPReLU(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.conv5 = _ConvWithPReLU(in_channels=128, out_channels=c, kernel_size=2, padding=0)
        self.norm = self._normlizationLayer(P=P)

    @staticmethod
    def _normlizationLayer(P=1):
        def _inner(z_hat: torch.Tensor):
            if z_hat.dim() == 4:
                batch_size = z_hat.size()[0]
                # k = np.prod(z_hat.size()[1:])
                k = torch.prod(torch.tensor(z_hat.size()[1:]))
            elif z_hat.dim() == 3:
                batch_size = 1
                # k = np.prod(z_hat.size())
                k = torch.prod(torch.tensor(z_hat.size()))
            else:
                raise Exception('Unknown size of input')
            # k = torch.tensor(k)
            z_temp = z_hat.reshape(batch_size, 1, 1, -1)
            z_trans = z_hat.reshape(batch_size, 1, -1, 1)
            tensor = torch.sqrt(P * k) * z_hat / torch.sqrt((z_temp @ z_trans))
            if batch_size == 1:
                return tensor.squeeze(0)
            return tensor

        return _inner

    def forward(self, x):
        _if_print(x.shape)
        # x = self.imgae_normalization(x)
        x = self.conv1(x)
        _if_print(x.shape)
        x = self.conv2(x)
        _if_print(x.shape)
        x = self.conv3(x)
        _if_print(x.shape)
        x = self.conv4(x)
        _if_print(x.shape)
        if not self.is_temp:
            x = self.conv5(x)
            _if_print(x.shape)
            x = self.norm(x)
            # exit(0)
        return x


class _Decoder(nn.Module):
    def __init__(self, c=1):
        super(_Decoder, self).__init__()
        # self.imgae_normalization = _image_normalization(norm_type='denormalization')
        self.tconv1 = _TransConvWithPReLU(
            in_channels=c, out_channels=128, kernel_size=2, stride=1, padding=0)
        self.tconv2 = _TransConvWithPReLU(
            in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.tconv3 = _TransConvWithPReLU(
            in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.tconv4 = _TransConvWithPReLU(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.tconv5 = _TransConvWithPReLU(
            in_channels=32, out_channels=3, kernel_size=4, stride=2, padding=1, activate=nn.Sigmoid())
        # may be some problems in tconv4 and tconv5, the kernal_size is not the same as the paper which is 5

    def forward(self, x):
        _if_print(x.shape)
        x = self.tconv1(x)
        _if_print(x.shape)
        x = self.tconv2(x)
        _if_print(x.shape)
        x = self.tconv3(x)
        _if_print(x.shape)
        x = self.tconv4(x)
        _if_print(x.shape)
        x = self.tconv5(x)
        _if_print(x.shape)
        # exit(0)
        # x = self.imgae_normalization(x)
        return x


class DeepJSCC(nn.Module):
    def __init__(self, c, channel_type='AWGN', snr=20):
        super(DeepJSCC, self).__init__()
        self.encoder = _Encoder(c=c)
        self.channel = channel.channel(channel_type, snr)
        self.decoder = _Decoder(c=c)

    def forward(self, x):
        z = self.encoder(x)
        z = self.channel(z)
        x_hat = self.decoder(z)
        return x_hat

    def change_channel(self, channel_type, snr):
        self.channel = channel.channel(channel_type, snr)
