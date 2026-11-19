import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    def __init__(self, Fg, Fl, Fint):
        super().__init__()

        self.Wg = nn.Sequential(
            nn.Conv3d(Fg, Fint, kernel_size=1),
            nn.BatchNorm3d(Fint)
        )

        self.Wx = nn.Sequential(
            nn.Conv3d(Fl, Fint, kernel_size=1),
            nn.BatchNorm3d(Fint)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(Fint, 1, kernel_size=1),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):

        g1 = self.Wg(g)
        x1 = self.Wx(x)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class AttentionUNet3D(nn.Module):

    def __init__(self, in_channels=1, num_classes=2, base_ch=16):
        super().__init__()

        self.enc1 = DoubleConv(in_channels, base_ch)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = DoubleConv(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = DoubleConv(base_ch * 2, base_ch * 4)
        self.pool3 = nn.MaxPool3d(2)

        self.bottleneck = DoubleConv(base_ch * 4, base_ch * 8)

        self.up3 = nn.ConvTranspose3d(
            base_ch * 8,
            base_ch * 4,
            kernel_size=2,
            stride=2
        )

        self.att3 = AttentionGate(
            Fg=base_ch * 4,
            Fl=base_ch * 4,
            Fint=base_ch * 2
        )

        self.dec3 = DoubleConv(base_ch * 8, base_ch * 4)

        self.up2 = nn.ConvTranspose3d(
            base_ch * 4,
            base_ch * 2,
            kernel_size=2,
            stride=2
        )

        self.att2 = AttentionGate(
            Fg=base_ch * 2,
            Fl=base_ch * 2,
            Fint=base_ch
        )

        self.dec2 = DoubleConv(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose3d(
            base_ch * 2,
            base_ch,
            kernel_size=2,
            stride=2
        )

        self.att1 = AttentionGate(
            Fg=base_ch,
            Fl=base_ch,
            Fint=base_ch // 2
        )

        self.dec1 = DoubleConv(base_ch * 2, base_ch)

        self.out_conv = nn.Conv3d(base_ch, num_classes, 1)

    def forward(self, x):

        e1 = self.enc1(x)

        e2 = self.enc2(self.pool1(e1))

        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        e3_att = self.att3(d3, e3)
        d3 = torch.cat([d3, e3_att], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        e2_att = self.att2(d2, e2)
        d2 = torch.cat([d2, e2_att], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        e1_att = self.att1(d1, e1)
        d1 = torch.cat([d1, e1_att], dim=1)
        d1 = self.dec1(d1)

        return self.out_conv(d1)


if __name__ == "__main__":

    x = torch.randn(2,1,96,96,96)

    model = AttentionUNet3D()

    y = model(x)

    print("Input :", x.shape)
    print("Output:", y.shape)
