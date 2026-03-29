import torch
import torch.nn as nn

class HSigmoid(nn.Module):
    def forward(self, x):
        return nn.functional.relu6(x + 3) / 6

class HSwish(nn.Module):
    def forward(self, x):
        return x * nn.functional.relu6(x + 3) / 6

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        reduced_channels = max(1, in_channels // reduction)
        self.fc1 = nn.Linear(in_channels, reduced_channels)
        self.fc2 = nn.Linear(reduced_channels, in_channels)
        self.relu = nn.ReLU()
        self.hsigmoid = HSigmoid()
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = x.mean([2, 3])
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.hsigmoid(y).view(b, c, 1, 1)
        return x * y

class MobileNetV3Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, use_se=True, use_hs=True):
        super().__init__()
        # Ensure expand_ratio is integer
        hidden_dim = int(in_channels * expand_ratio)
        self.use_residual = stride == 1 and in_channels == out_channels
        
        layers = []
        
        # Expansion phase
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, 1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(HSwish() if use_hs else nn.ReLU(inplace=True))
        
        # Depthwise convolution
        layers.append(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, 
                      kernel_size // 2, groups=hidden_dim, bias=False)
        )
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(HSwish() if use_hs else nn.ReLU(inplace=True))
        
        # Squeeze-and-excitation
        if use_se:
            layers.append(SqueezeExcitation(hidden_dim))
        
        # Projection phase
        layers.append(nn.Conv2d(hidden_dim, out_channels, 1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV3Small(nn.Module):
    """MobileNetV3-Small for CIFAR-10"""
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            HSwish()
        )
        
        # Config: (out_channels, kernel_size, stride, expand_ratio, use_se, use_hs)
        # Using integer expand_ratio
        self.blocks = nn.Sequential(
            MobileNetV3Block(16, 16, 3, 2, 1, use_se=True, use_hs=False),
            MobileNetV3Block(16, 24, 3, 2, 3, use_se=False, use_hs=False),  # 2.5 -> 3
            MobileNetV3Block(24, 24, 3, 1, 3, use_se=False, use_hs=False),  # 3.5 -> 3
            MobileNetV3Block(24, 40, 5, 2, 4, use_se=True, use_hs=True),
            MobileNetV3Block(40, 40, 5, 1, 5, use_se=True, use_hs=True),
            MobileNetV3Block(40, 48, 5, 1, 5, use_se=True, use_hs=True),
            MobileNetV3Block(48, 48, 5, 1, 5, use_se=True, use_hs=True),
            MobileNetV3Block(48, 96, 5, 2, 6, use_se=True, use_hs=True),
            MobileNetV3Block(96, 96, 5, 1, 6, use_se=True, use_hs=True),
            MobileNetV3Block(96, 96, 5, 1, 6, use_se=True, use_hs=True),
        )
        
        self.head = nn.Sequential(
            nn.Conv2d(96, 576, 1, bias=False),
            nn.BatchNorm2d(576),
            HSwish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(576, num_classes)
        )
        
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x