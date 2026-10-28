"""U-Net for semantic segmentation of Mars terrain (1-channel input, 4 classes)"""

import torch
from torch import nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels): # creates the layers
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False), # 3×3 kernel
            nn.BatchNorm2d(num_features=out_channels), # keeps one mean/std (and one learned scale and shift) per channel
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels,kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        )

    def forward(self, x): # passes x through the sequence and returns the result
        return self.block(x)

    
# UNet uses DoubleConv blocks, max-pooling, transposed convs and the final 1×1 conv
class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=4): # create the layers 
        super().__init__()
        # input has 1 channel and size 512 
        self.down1 = DoubleConv(in_channels, 64) # 64x512
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2) # one pool layer, reused, no weights 

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(1024, 512) # 1024 in because of the concat
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3 =  DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x): # connect the layers, store down the outputs to reuse them as skips 
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        d4 = self.down4(self.pool(d3))

        b = self.bottleneck(self.pool(d4))

        u4 = self.up4(b)  # 1. upsample
        u4 = self.conv4(torch.cat([u4, d4], dim=1))  # 2. concat + DoubleConv
        u3 = self.up3(u4)
        u3 = self.conv3(torch.cat([u3, d3], dim=1))
        u2 = self.up2(u3)
        u2 = self.conv2(torch.cat([u2, d2], dim=1))
        u1 = self.up1(u2)
        u1 = self.conv1(torch.cat([u1, d1], dim=1))

        return self.out(u1)