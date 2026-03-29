import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """LayerNorm for ConvNeXt"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
    
    def forward(self, x):
        # x shape: (B, C, H, W) -> (B, H, W, C) for LayerNorm
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.weight * x + self.bias
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        return x


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = nn.Identity()  # Simplified for now
        
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm(x.permute(0, 3, 1, 2))  # Fix shape
        x = x.permute(0, 2, 3, 1)  # Back to (B, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        x = input + self.drop_path(x)
        return x


class ConvNeXtTiny(nn.Module):
    """ConvNeXt-Tiny for CIFAR-10"""
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Simplified stem for CIFAR-10 (32x32 input)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )
        
        # Simplified stages
        self.stage1 = self._make_stage(64, 64, depth=2)
        self.stage2 = self._make_stage(64, 128, depth=2, stride=2)
        self.stage3 = self._make_stage(128, 256, depth=4, stride=2)
        self.stage4 = self._make_stage(256, 512, depth=2, stride=2)
        
        self.norm = nn.BatchNorm2d(512)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(512, num_classes)
        
    def _make_stage(self, in_channels, out_channels, depth, stride=1):
        layers = []
        
        # Downsample if needed
        if stride > 1 or in_channels != out_channels:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
            )
            in_channels = out_channels
        
        # ConvNeXt blocks
        for i in range(depth):
            layers.append(ConvNeXtBlock(out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stem(x)
        
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x