from typing import Optional, Sequence
import torch
import torch.nn as nn
from thop import profile
import torch.nn.functional as F
import math
from pytorch_wavelets import DWTForward

class Dwt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Dwt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
                                    nn.Conv2d(in_ch*4, out_ch, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(out_ch),
                                    nn.ReLU(inplace=True),
                                    )
    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.01, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
class Partial_conv(nn.Module):
    def __init__(self, dim, n_div=4):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)
        self.forward = self.forward_split_cat

    def forward_split_cat(self, x):
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x


class DCEA(nn.Module):
    def __init__(self, in_channel, decay=2,global_ratio=0.7):
        super(DCEA, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // decay, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // decay, in_channel, 1),
            nn.Sigmoid()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // decay, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // decay, in_channel, 1),
            nn.Sigmoid()
        )
        self.gpool = nn.AdaptiveAvgPool2d(1)
        self.gapool = nn.AdaptiveMaxPool2d(1)
        self.local_conv = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=0)
        self.global_ratio = global_ratio
        self.local_ratio = 1 - global_ratio
    def forward(self, x):
        local_pool = nn.AdaptiveAvgPool2d(3)(x)
        local_feat = self.local_conv(local_pool)
        local_se = self.layer1(local_feat.mean(dim=[2, 3], keepdim=True))
        gp = self.gpool(x)
        global_se = self.layer1(gp)
        se = self.global_ratio * global_se + self.local_ratio * local_se
        x = x * se
        gap = self.gapool(x)
        se2 = self.layer2(gap)
        return x * se2


class EncoderConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,global_ratio=0.8):
        super(EncoderConvBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.partial_conv = Partial_conv(out_channels, n_div=4)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.LeakyReLU(0.01, inplace=True)
        decay = 4 if out_channels >= 256 else 2
        self.dse = DCEA(out_channels, decay=decay, global_ratio=global_ratio)

    def forward(self, x):
        x = self.conv1(x)
        x = self.partial_conv(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dse(x)
        return x



class DecoderConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,global_ratio=0.5):
        super(DecoderConvBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.01, inplace=True)
        )

        self.partial_conv = Partial_conv(out_channels, n_div=4)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.LeakyReLU(0.01, inplace=True)
        decay = 4 if out_channels >= 256 else 2  # 动态压缩率
        self.dse = DCEA(out_channels, decay=decay, global_ratio=global_ratio)
    def forward(self, x):
        x = self.conv1(x)
        x = self.partial_conv(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dse(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self, channels):
        super(Bottleneck, self).__init__()
        self.partial_conv1 = Partial_conv(channels, n_div=2)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.partial_conv2 = Partial_conv(channels, n_div=4)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        x = self.relu1(self.bn1(self.partial_conv1(x)))
        x = self.bn2(self.partial_conv2(x))
        x = x + identity
        return x

class Conv_Extra(nn.Module):
    def __init__(self, channel, norm_layer, act_layer):
        super(Conv_Extra, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(channel, 64, 1),
                                   nn.BatchNorm2d(64),
                                   act_layer(),
                                   nn.Conv2d(64, 64, 3, stride=1, padding=1, dilation=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   act_layer(),
                                   nn.Conv2d(64, channel, 1),
                                   nn.BatchNorm2d(channel))

    def forward(self, x):
        out = self.block(x)
        return out


class GGFR(nn.Module):
    def __init__(self, dim, size, sigma, norm_layer, act_layer, feature_extra=True):
        super().__init__()
        self.feature_extra = feature_extra
        gaussian = self.gaussian_kernel(size, sigma)
        gaussian = nn.Parameter(data=gaussian, requires_grad=False).clone()
        self.gaussian = nn.Conv2d(dim, dim, kernel_size=size, stride=1, padding=int(size // 2), groups=dim, bias=False)
        self.gaussian.weight.data = gaussian.repeat(dim, 1, 1, 1)
        self.gaussian.weight.requires_grad = False
        self.norm = nn.BatchNorm2d(dim)
        self.act = act_layer()
        if feature_extra == True:
            self.conv_extra = Conv_Extra(dim, norm_layer,act_layer)
        self.partial_refine = Partial_conv(dim, n_div=4)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        edges_o = self.gaussian(x)
        gaussian = self.act(self.norm(edges_o))
        if self.feature_extra == True:
            out = self.conv_extra(x + gaussian)
        else:
            out = gaussian
        out = out + self.relu(self.bn(self.partial_refine(out)))
        return out

    def gaussian_kernel(self, size: int, sigma: float):
        kernel = torch.FloatTensor([
            [(1 / (2 * math.pi * sigma ** 2)) * math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
             for x in range(-size // 2 + 1, size // 2 + 1)]
             for y in range(-size // 2 + 1, size // 2 + 1)
        ]).unsqueeze(0).unsqueeze(0)
        return kernel / kernel.sum()

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DecoderConvBlock(in_channels, out_channels)
        self.ega = GGFR(in_channels // 2,size=5, sigma=1.0,
                                norm_layer = dict(type='BN', requires_grad=True),
                                act_layer=nn.ReLU,
                                )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        x = self.ega(x)
        return x


class PFRUNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1):
        super(PFRUNet, self).__init__()
        self.inc = EncoderConvBlock(in_channels, 64)
        self.down1 = nn.Sequential(
            Dwt(64, 128),
            EncoderConvBlock(128, 128)
        )

        self.down2 = nn.Sequential(
            Dwt(128, 256),
            EncoderConvBlock(256, 256)
        )

        self.down3 = nn.Sequential(nn.MaxPool2d(2), EncoderConvBlock(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), EncoderConvBlock(512, 1024))
        # self.bottleneck =Bottleneck(1024)
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
        x5 = self.down4(x4)
        # x5 = self.bottleneck(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

def model(in_channels=1, out_channels=1):
    return PFRUNet(in_channels=in_channels, out_channels=out_channels)


if __name__ == '__main__':
    test_input = torch.randn(1, 1, 256, 256)
    model = model(in_channels=1, out_channels=1)
    output = model(test_input)
    print(f"\n输入尺寸: {test_input.shape}")
    print(f"输出尺寸: {output.shape}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params / 1e6:.2f} M")
    flops, params = profile(model, (test_input,))
    print("-" * 50)
    print('FLOPs = ' + str(flops / 1000 ** 3) + ' G')
    print('Params = ' + str(params / 1000 ** 2) + ' M')
