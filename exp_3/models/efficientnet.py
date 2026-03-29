import torch
import torch.nn as nn

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        reduced_channels = max(1, in_channels // reduction)
        self.fc1 = nn.Linear(in_channels, reduced_channels)
        self.fc2 = nn.Linear(reduced_channels, in_channels)
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = x.mean([2, 3])
        y = self.fc1(y)
        y = torch.relu(y)
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y


class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, use_se=True):
        super().__init__()
        hidden_dim = int(in_channels * expand_ratio)
        self.use_residual = stride == 1 and in_channels == out_channels
        
        layers = []
        
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, 1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.SiLU())
        
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, 
                      kernel_size // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
        ])
        
        if use_se:
            layers.append(SqueezeExcitation(hidden_dim))
        
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class EfficientNetB0(nn.Module):
    """EfficientNet-B0 for CIFAR-10"""
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU()
        )
        
        # Config: (out_channels, kernel_size, stride, expand_ratio, use_se)
        self.blocks = nn.Sequential(
            MBConv(32, 16, 3, 1, 1, use_se=False),
            MBConv(16, 24, 3, 2, 6, use_se=False),
            MBConv(24, 24, 3, 1, 6, use_se=False),
            MBConv(24, 40, 5, 2, 6, use_se=True),
            MBConv(40, 40, 5, 1, 6, use_se=True),
            MBConv(40, 80, 3, 2, 6, use_se=True),
            MBConv(80, 80, 3, 1, 6, use_se=True),
            MBConv(80, 112, 5, 1, 6, use_se=True),
            MBConv(112, 112, 5, 1, 6, use_se=True),
            MBConv(112, 192, 5, 2, 6, use_se=True),
            MBConv(192, 192, 5, 1, 6, use_se=True),
            MBConv(192, 320, 3, 1, 6, use_se=True),
        )
        
        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )
        
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x