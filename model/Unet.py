import torch
import torch.nn as nn
from thop import profile
import math
from pytorch_wavelets import DWTForward

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            # nn.GroupNorm(32,out_channels),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            # nn.GroupNorm(32, out_channels),
            nn.LeakyReLU(0.01, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UpBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class StandardUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(StandardUNet, self).__init__()

        self.inc = ConvBlock(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(256, 512))
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), ConvBlock(512, 1024))

        self.up1 = UpBlock(1024, 512)
        self.up2 = UpBlock(512, 256)
        self.up3 = UpBlock(256, 128)
        self.up4 = UpBlock(128, 64)
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)
        self.out_activation = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits

def UNet(in_channels=1, out_channels=1):
    return StandardUNet(in_channels=in_channels, out_channels=out_channels)


if __name__ == '__main__':
    test_input = torch.randn(2, 3, 384, 384)
    model = UNet(in_channels=3, out_channels=1)
    output = model(test_input)
  