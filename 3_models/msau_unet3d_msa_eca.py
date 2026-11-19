import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ECA3D(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(
            1, 1,
            kernel_size=k_size,
            padding=(k_size - 1) // 2,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class SpatialAttention3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(
            2, 1,
            kernel_size=7,
            padding=3,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.sigmoid(self.conv(y))
        return x * y


class MultiScaleAttentionBlock3D(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.Conv3d(channels, channels, 1, padding=0, bias=False),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
        )

        self.branch3 = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
        )

        self.branch5 = nn.Sequential(
            nn.Conv3d(channels, channels, 5, padding=2, bias=False),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
        )

        self.branch_dilated = nn.Sequential(
            nn.Conv3d(
                channels, channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                bias=False
            ),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
        )

        self.fuse = nn.Sequential(
            nn.Conv3d(channels * 4, channels, 1, bias=False),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
        )

        self.eca = ECA3D(channels)
        self.spatial = nn.Identity()

    def forward(self, x):
        identity = x

        x1 = self.branch1(x)
        x3 = self.branch3(x)
        x5 = self.branch5(x)
        xd = self.branch_dilated(x)

        out = torch.cat([x1, x3, x5, xd], dim=1)
        out = self.fuse(out)
        out = self.eca(out)
        out = self.spatial(out)

        return out + identity


class MSAUNet3DMSAECA(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, base_ch=16):
        super().__init__()

        self.enc1 = DoubleConv(in_channels, base_ch)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = DoubleConv(base_ch, base_ch * 2)
        self.msa2 = MultiScaleAttentionBlock3D(base_ch * 2)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = DoubleConv(base_ch * 2, base_ch * 4)
        self.msa3 = MultiScaleAttentionBlock3D(base_ch * 4)
        self.pool3 = nn.MaxPool3d(2)

        self.bottleneck = DoubleConv(base_ch * 4, base_ch * 8)
        self.msa4 = MultiScaleAttentionBlock3D(base_ch * 8)

        self.up3 = nn.ConvTranspose3d(
            base_ch * 8,
            base_ch * 4,
            kernel_size=2,
            stride=2
        )
        self.dec3 = DoubleConv(base_ch * 8, base_ch * 4)

        self.up2 = nn.ConvTranspose3d(
            base_ch * 4,
            base_ch * 2,
            kernel_size=2,
            stride=2
        )
        self.dec2 = DoubleConv(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose3d(
            base_ch * 2,
            base_ch,
            kernel_size=2,
            stride=2
        )
        self.dec1 = DoubleConv(base_ch * 2, base_ch)

        self.out_conv = nn.Conv3d(base_ch, num_classes, 1)

    def crop_or_pad_like(self, src, ref):
        diff_d = ref.size(2) - src.size(2)
        diff_h = ref.size(3) - src.size(3)
        diff_w = ref.size(4) - src.size(4)

        src = F.pad(
            src,
            [
                diff_w // 2,
                diff_w - diff_w // 2,
                diff_h // 2,
                diff_h - diff_h // 2,
                diff_d // 2,
                diff_d - diff_d // 2,
            ]
        )

        return src

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.msa2(self.enc2(p1))
        p2 = self.pool2(e2)

        e3 = self.msa3(self.enc3(p2))
        p3 = self.pool3(e3)

        b = self.msa4(self.bottleneck(p3))

        d3 = self.up3(b)
        d3 = self.crop_or_pad_like(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.crop_or_pad_like(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.crop_or_pad_like(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out_conv(d1)


if __name__ == "__main__":
    x = torch.randn(1, 1, 64, 96, 96)
    model = MSAUNet3DMSAECA(in_channels=1, num_classes=2, base_ch=16)
    y = model(x)
    print(y.shape)
