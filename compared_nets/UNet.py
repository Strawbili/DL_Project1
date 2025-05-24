import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    Two consecutive convolutional layers with BatchNorm and ReLU.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """
    Downscaling with MaxPool followed by DoubleConv.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.pool_conv(x)

class Up(nn.Module):
    """
    Upscaling then DoubleConv. Uses ConvTranspose2d for upsampling.
    """
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        if bilinear:
            # Use the normal convolutions to reduce the number of channels
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            # ConvTranspose2d
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is BCHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # concatenate along the channels axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNetClassifier(nn.Module):
    """
    UNet-based classification network. Supports arbitrary input size; here tested on 1x64x48 inputs.
    """
    def __init__(self, in_channels=1, num_classes=62, bilinear=False):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)

        # bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # upsampling path (not used for classification skip if desired)
        self.up4 = Up(1024, 512, bilinear)
        self.up3 = Up(512, 256, bilinear)
        self.up2 = Up(256, 128, bilinear)
        self.up1 = Up(128, 64, bilinear)

        # classification head: reduce to num_classes channels
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)
        # global average pooling to get [B, num_classes, 1, 1]
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Bottleneck
        x6 = self.bottleneck(x5)

        # Decoder (skip if classification from bottleneck is preferred)
        x = self.up4(x6, x5)
        x = self.up3(x, x4)
        x = self.up2(x, x3)
        x = self.up1(x, x2)

        # Classification head
        x = self.classifier(x)                # [B, num_classes, H, W]
        x = self.global_pool(x)               # [B, num_classes, 1, 1]
        x = x.view(x.size(0), self.num_classes)
        return x

if __name__ == '__main__':
    # Quick sanity check
    model = UNetClassifier(in_channels=1, num_classes=62)
    x = torch.randn(4, 1, 64, 48)
    y = model(x)
    print(y.shape)  # Expected: [4, 62]
