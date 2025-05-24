import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    两层 Conv → BN → ReLU，最后可选 Dropout2d
    """
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """
    MaxPool → DoubleConv → Dropout2d
    """
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout=dropout),
        )

    def forward(self, x):
        return self.pool_conv(x)

class Up(nn.Module):
    """
    上采样（ConvTranspose2d）→ 拼接 → DoubleConv → Dropout2d
    """
    def __init__(self, in_channels, out_channels, dropout=0.0, bilinear=False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # 注意：channels 拼接后为 in_channels + skip_channels，需要传入正确值
            self.conv = DoubleConv(in_channels, out_channels, dropout=dropout)
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(out_channels * 2, out_channels, dropout=dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 对齐尺寸
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX//2, diffX - diffX//2,
                        diffY//2, diffY - diffY//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNetClassifier(nn.Module):
    """
    带 Dropout 的 UNet 分类网络
    """
    def __init__(self, in_channels=1, num_classes=62, bilinear=False):
        super().__init__()
        # 我们给不同层设置不同的 dropout 比例
        self.inc    = DoubleConv(in_channels,  32,  dropout=0.1)
        self.down1  = Down(32,  64,  dropout=0.1)
        self.down2  = Down(64, 128,  dropout=0.1)
        self.down3  = Down(128,256,  dropout=0.1)
        self.down4  = Down(256,512,  dropout=0.1)

        # bottleneck 加大 dropout
        self.bottleneck = nn.Sequential(
            DoubleConv(512, 1024, dropout=0.5),
        )

        # 上采样路径
        self.up4 = Up(1024, 512, dropout=0.1, bilinear=bilinear)
        self.up3 = Up(512,  256, dropout=0.1, bilinear=bilinear)
        self.up2 = Up(256,  128, dropout=0.1, bilinear=bilinear)
        self.up1 = Up(128,   64, dropout=0.1, bilinear=bilinear)

        # 分类头
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)     # [B, 32, H, W]
        x2 = self.down1(x1)  # [B, 64, H/2, W/2]
        x3 = self.down2(x2)  # [B,128, H/4, W/4]
        x4 = self.down3(x3)  # [B,256, H/8, W/8]
        x5 = self.down4(x4)  # [B,512, H/16,W/16]

        # Bottleneck
        x6 = self.bottleneck(x5)

        # Decoder
        x = self.up4(x6, x5)
        x = self.up3(x,  x4)
        x = self.up2(x,  x3)
        x = self.up1(x,  x2)

        # Classification
        x = self.classifier(x)      # [B, num_classes, H, W]
        x = self.global_pool(x)     # [B, num_classes, 1, 1]
        return x.view(x.size(0), -1)

if __name__ == '__main__':
    # Quick sanity check
    model = UNetClassifier(in_channels=1, num_classes=62)
    x = torch.randn(4, 1, 64, 48)
    y = model(x)
    print(y.shape)  # Expected: [4, 62]