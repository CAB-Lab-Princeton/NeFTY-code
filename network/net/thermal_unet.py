# 3D U-Net used by the U-Net baselines.
#
# The building blocks DoubleConv, Down, Up and OutConv are adapted (2D -> 3D) from
# Pytorch-UNet, https://github.com/milesial/Pytorch-UNet
# Copyright (c) milesial, licensed under the GNU General Public License v3.0.
#
# Accordingly, this file is distributed under the GNU General Public License v3.0
# (see LICENSES/GPL-3.0.txt), not under the MIT License that covers the rest of
# this repository.
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()

        # if trilinear, use the normal upsampling
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffZ = x2.size(2) - x1.size(2)
        diffY = x2.size(3) - x1.size(3)
        diffX = x2.size(4) - x1.size(4)

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class ThermalUNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True, min_diffusivity=0.003, max_diffusivity=0.25):
        super(ThermalUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.min_diffusivity = min_diffusivity
        self.max_diffusivity = max_diffusivity

        # Initial signal processing:
        # Input is (B, 1, 101, 64, 64). We need to map 101 -> 16 for standard 3D Unet depth handling
        # We can use interpolation in forward or a dedicated conv layer.
        
        # Architecture
        # factor = 2 if bilinear else 1
        factor = 1 # Keep simple
        
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        # self.down4 = Down(256, 512 // factor)
        
        # self.up1 = Up(512, 256 // factor, bilinear)
        self.up1 = Up(256 + 128, 128, bilinear) # Skip from down2
        self.up2 = Up(128 + 64, 64, bilinear)   # Skip from down1
        self.up3 = Up(64 + 32, 32, bilinear)    # Skip from inc
        
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        """
        x: (B, 1, T=101, H, W)
        """
        # Interpolate Time dimension to match Z dimension (16)
        # Target shape: (B, 1, 16, H, W)
        target_z = 16
        if x.size(2) != target_z:
             x = F.interpolate(x, size=(target_z, x.size(3), x.size(4)), mode='trilinear', align_corners=True)
        
        x1 = self.inc(x)         # 32, 16, 64, 64
        x2 = self.down1(x1)      # 64, 8, 32, 32
        x3 = self.down2(x2)      # 128, 4, 16, 16
        x4 = self.down3(x3)      # 256, 2, 8, 8
        
        # Decoder
        x = self.up1(x4, x3)     # 128, 4, 16, 16
        x = self.up2(x, x2)      # 64, 8, 32, 32
        x = self.up3(x, x1)      # 32, 16, 64, 64
        
        logits = self.outc(x)    # 1, 16, 64, 64
        
        # Sigmoid scaling to diffusivity range
        out = torch.sigmoid(logits) * (self.max_diffusivity - self.min_diffusivity) + self.min_diffusivity
        
        return out
