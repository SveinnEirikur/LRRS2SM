from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange

class UNet_s2(nn.Module):

    def __init__(self, in_channels=12, out_channels=12, init_features=32, sub_channels=None, init_slope=1.0):
        if sub_channels is None:
            sub_channels = [6, 2]
        super(UNet_s2, self).__init__()

        features = init_features
        self.encoder1 = UNet_s2._block(in_channels, features, init_slope, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet_s2._block(features, features * 2, init_slope, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.encoder3 = UNet_s2._block(features * 2, features * 4, init_slope, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet_s2._block(features * 4, features * 8, init_slope, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=3)

        self.bottleneck = UNet_s2._block(features * 8, features * 18 * 2, init_slope, name="bottleneck")

        self.upshuff4 = Rearrange("b (h2 w2 c) h w -> b c (h h2) (w w2)", h2=3, w2=3)
        self.upconv4 = nn.Conv2d(
            in_channels=features * 4,
            out_channels=features * 8,
            kernel_size=5,
            padding=2,
            bias=False,
        )

        self.decoder4 = UNet_s2._block(features * 16, features * 8, init_slope, name="dec4")
        self.upshuff3 = Rearrange("b (c h2 w2) h w -> b c (h h2) (w w2)", h2=2, w2=2)
        self.upconv3 = nn.Conv2d(
            in_channels=features * 2,
            out_channels=features * 4,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        self.decoder3 = UNet_s2._block(features * 8, features * 9, init_slope, name="dec3")
        self.upshuff2 = Rearrange("b (c h2 w2) h w -> b c (h h2) (w w2)", h2=3, w2=3)
        self.upconv2 = nn.Conv2d(
            in_channels=features,
            out_channels=features * 2,
            kernel_size=5,
            padding=2,
            bias=False,
        )

        self.decoder2 = UNet_s2._block(features * 4, features * 2, init_slope, name="dec2")
        self.upshuff1 = Rearrange("b (c h2 w2) h w -> b c (h h2) (w w2)", h2=2, w2=2)
        self.upconv1 = nn.Conv2d(
            in_channels=features // 2,
            out_channels=features,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        self.decoder1 = UNet_s2._block(features * 2, features, init_slope, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1, bias=False,
        )

        self.activation = nn.PReLU(num_parameters=out_channels, init=init_slope)

    def forward(self, x):
        in_shape = x.shape
        w_pad_shape = 36*np.ceil(x.shape[-2]/36)-x.shape[-2]
        w_pad_shape = (int(np.floor(w_pad_shape/2)), int(np.ceil(w_pad_shape/2)))
        h_pad_shape = 36*np.ceil(x.shape[-1]/36)-x.shape[-1]
        h_pad_shape = (int(np.floor(h_pad_shape/2)), int(np.ceil(h_pad_shape/2)))

        x = rearrange(_fw_symm_pad(rearrange(x, "... w h -> ... h w"), w_pad_shape), "... h w -> ... w h")
        x = _fw_symm_pad(x, h_pad_shape)

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upshuff4(bottleneck)
        dec4 = self.upconv4(dec4)
        dec4 = torch.cat((dec4, enc4*0.5**3), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upshuff3(dec4)
        dec3 = self.upconv3(dec3)
        dec3 = torch.cat((dec3, enc3*0.5**2), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upshuff2(dec3)
        dec2 = self.upconv2(dec2)
        dec2 = torch.cat((dec2, enc2*0.5**1), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upshuff1(dec2)
        dec1 = self.upconv1(dec1)
        dec1 = torch.cat((dec1, enc1*0.5**0), dim=1)
        dec1 = self.decoder1(dec1)
        dec1 = dec1[:,
                    :,
                    w_pad_shape[0]:None if w_pad_shape[1] == 0 else -w_pad_shape[1],
                    h_pad_shape[0]:None if h_pad_shape[1] == 0 else -h_pad_shape[1]]
        assert(dec1.shape[-2]==in_shape[-2] and dec1.shape[-1]==in_shape[-1])
        return self.activation(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, init_slope, name, kernel_size=3, padding=1):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=kernel_size,
                            padding=padding,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.InstanceNorm2d(num_features=features)),
                    (name + "prelu1", nn.PReLU(num_parameters=features, init=init_slope)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=kernel_size,
                            padding=padding,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.InstanceNorm2d(num_features=features)),
                    (name + "prelu2", nn.PReLU(num_parameters=features, init=init_slope)),
                ],
            ),
        )


class UNet_l8(nn.Module):

    def __init__(self, in_channels=12, out_channels=12, init_features=32, sub_channels=None, init_slope=1.0):
        if sub_channels is None:
            sub_channels = [6, 2]
        super(UNet_l8, self).__init__()

        features = init_features
        self.encoder1 = UNet_s2._block(in_channels, features, init_slope, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.encoder2 = UNet_s2._block(features, features * 3, init_slope, name="enc2") 
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet_s2._block(features * 3, features * 6, init_slope, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.encoder4 = UNet_s2._block(features * 6, features * 12, init_slope, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet_s2._block(features * 12, features * 36, init_slope, name="bottleneck")

        self.upshuff4 = Rearrange("b (h2 w2 c) h w -> b c (h h2) (w w2)", h2=2, w2=2)
        self.upconv4 = nn.Conv2d(
            in_channels=features * 9,
            out_channels=features * 12,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        self.decoder4 = UNet_s2._block(features * 24, features * 18, init_slope, name="dec4")
        self.upshuff3 = Rearrange("b (c h2 w2) h w -> b c (h h2) (w w2)", h2=3, w2=3)
        self.upconv3 = nn.Conv2d(
            in_channels=features * 2,
            out_channels=features * 6,
            kernel_size=5,
            padding=2,
            bias=False,
        )

        self.decoder3 = UNet_s2._block(features * 12, features * 12, init_slope, name="dec3")
        self.upshuff2 = Rearrange("b (c h2 w2) h w -> b c (h h2) (w w2)", h2=2, w2=2)
        self.upconv2 = nn.Conv2d(
            in_channels=features * 3,
            out_channels=features * 3,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        self.decoder2 = UNet_s2._block(features * 6, features * 9, init_slope, name="dec2")
        self.upshuff1 = Rearrange("b (c h2 w2) h w -> b c (h h2) (w w2)", h2=3, w2=3)
        self.upconv1 = nn.Conv2d(
            in_channels=features,
            out_channels=features,
            kernel_size=5,
            padding=2,
            bias=False,
        )

        self.decoder1 = UNet_s2._block(features * 2, features, init_slope, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1, bias=False,
        )

        self.activation = nn.PReLU(num_parameters=out_channels, init=init_slope)

    def forward(self, x):
        in_shape = x.shape
        w_pad_shape = 36*np.ceil(x.shape[-2]/36)-x.shape[-2]
        w_pad_shape = (int(np.floor(w_pad_shape/2)), int(np.ceil(w_pad_shape/2)))
        h_pad_shape = 36*np.ceil(x.shape[-1]/36)-x.shape[-1]
        h_pad_shape = (int(np.floor(h_pad_shape/2)), int(np.ceil(h_pad_shape/2)))

        x = rearrange(_fw_symm_pad(rearrange(x, "... w h -> ... h w"), w_pad_shape), "... h w -> ... w h")
        x = _fw_symm_pad(x, h_pad_shape)

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upshuff4(bottleneck)
        dec4 = self.upconv4(dec4)
        dec4 = torch.cat((dec4, enc4*0.5**3), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upshuff3(dec4)
        dec3 = self.upconv3(dec3)
        dec3 = torch.cat((dec3, enc3*0.5**2), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upshuff2(dec3)
        dec2 = self.upconv2(dec2)
        dec2 = torch.cat((dec2, enc2*0.5**1), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upshuff1(dec2)
        dec1 = self.upconv1(dec1)
        dec1 = torch.cat((dec1, enc1*0.5**0), dim=1)
        dec1 = self.decoder1(dec1)
        dec1 = dec1[:,
                    :,
                    w_pad_shape[0]:None if w_pad_shape[1] == 0 else -w_pad_shape[1],
                    h_pad_shape[0]:None if h_pad_shape[1] == 0 else -h_pad_shape[1]]
        assert(dec1.shape[-2]==in_shape[-2] and dec1.shape[-1]==in_shape[-1])
        return self.activation(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, init_slope, name, kernel_size=3, padding=1):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=kernel_size,
                            padding=padding,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.InstanceNorm2d(num_features=features)),
                    (name + "prelu1", nn.PReLU(num_parameters=features, init=init_slope)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=kernel_size,
                            padding=padding,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.InstanceNorm2d(num_features=features)),
                    (name + "prelu2", nn.PReLU(num_parameters=features, init=init_slope)),
                ],
            ),
        )


def _fw_symm_pad(im: torch.Tensor, padding: Tuple[int, int]):
    w = im.shape[-1]
    left, right = padding

    x_idx = np.arange(int(-left), int(w+right))

    def reflect(x, minx, maxx):
        """ Reflects an array around two points making a triangular waveform that ramps up
        and down,  allowing for pad lengths greater than the input length """
        rng = maxx - minx
        double_rng = 2*rng
        mod = np.fmod(x - minx, double_rng)
        normed_mod = np.where(mod < 0, mod+double_rng, mod)
        out = np.where(normed_mod >= rng, double_rng - normed_mod, normed_mod) + minx
        return np.array(out, dtype=x.dtype)

    x_pad = reflect(x_idx, -0.5, float(w-0.5))
    return im[..., x_pad]
