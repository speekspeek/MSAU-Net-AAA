import torch
import torch.nn as nn


class BasicBlock3D(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(
                in_ch,
                out_ch,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv3d(
                out_ch,
                out_ch,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):

        return self.block(x)



class SingleAttention3D(nn.Module):

    """
    Ablation:
    Remove multi-scale branches.
    Only single-scale convolution + attention.
    """

    def __init__(self, channels):

        super().__init__()


        self.feature = nn.Sequential(

            nn.Conv3d(
                channels,
                channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm3d(channels),

            nn.ReLU(inplace=True)
        )


        self.attention = nn.Sequential(

            nn.Conv3d(
                channels,
                1,
                kernel_size=1
            ),

            nn.Sigmoid()
        )


    def forward(self,x):

        feat = self.feature(x)

        weight = self.attention(feat)

        out = feat * weight + x

        return out




class MSAUNet3D_NoMultiScale(nn.Module):


    def __init__(
        self,
        in_channels=1,
        num_classes=2,
        base_ch=16
    ):

        super().__init__()


        # Encoder

        self.enc1 = BasicBlock3D(
            in_channels,
            base_ch
        )


        self.enc2 = BasicBlock3D(
            base_ch,
            base_ch*2
        )


        self.enc3 = BasicBlock3D(
            base_ch*2,
            base_ch*4
        )


        self.pool = nn.MaxPool3d(
            kernel_size=2
        )



        # ======================
        # No Multi-scale MSA
        # ======================

        self.msa2 = SingleAttention3D(
            base_ch*2
        )


        self.msa3 = SingleAttention3D(
            base_ch*4
        )



        self.bottleneck = nn.Sequential(

            BasicBlock3D(
                base_ch*4,
                base_ch*8
            ),

            SingleAttention3D(
                base_ch*8
            )
        )



        # Decoder

        self.up3 = nn.ConvTranspose3d(
            base_ch*8,
            base_ch*4,
            kernel_size=2,
            stride=2
        )


        self.dec3 = BasicBlock3D(
            base_ch*8,
            base_ch*4
        )



        self.up2 = nn.ConvTranspose3d(
            base_ch*4,
            base_ch*2,
            kernel_size=2,
            stride=2
        )


        self.dec2 = BasicBlock3D(
            base_ch*4,
            base_ch*2
        )



        self.up1 = nn.ConvTranspose3d(
            base_ch*2,
            base_ch,
            kernel_size=2,
            stride=2
        )


        self.dec1 = BasicBlock3D(
            base_ch*2,
            base_ch
        )



        self.out = nn.Conv3d(
            base_ch,
            num_classes,
            kernel_size=1
        )




    # ============================
    # 解决奇数尺寸 concat 问题
    # ============================

    def crop_concat(self, x, skip):


        diff_z = skip.size(2)-x.size(2)
        diff_y = skip.size(3)-x.size(3)
        diff_x = skip.size(4)-x.size(4)


        if diff_z !=0 or diff_y !=0 or diff_x !=0:


            z1 = diff_z//2
            y1 = diff_y//2
            x1 = diff_x//2


            skip = skip[
                :,
                :,
                z1:z1+x.size(2),
                y1:y1+x.size(3),
                x1:x1+x.size(4)
            ]


        return torch.cat(
            [x,skip],
            dim=1
        )




    def forward(self,x):


        # encoder

        e1 = self.enc1(x)


        p1 = self.pool(e1)



        e2 = self.enc2(p1)

        e2 = self.msa2(e2)


        p2 = self.pool(e2)



        e3 = self.enc3(p2)

        e3 = self.msa3(e3)


        p3 = self.pool(e3)



        b = self.bottleneck(p3)



        # decoder


        d3 = self.up3(b)

        d3 = self.crop_concat(
            d3,
            e3
        )

        d3 = self.dec3(d3)



        d2 = self.up2(d3)

        d2 = self.crop_concat(
            d2,
            e2
        )

        d2 = self.dec2(d2)



        d1 = self.up1(d2)

        d1 = self.crop_concat(
            d1,
            e1
        )

        d1 = self.dec1(d1)



        out = self.out(d1)


        return out
