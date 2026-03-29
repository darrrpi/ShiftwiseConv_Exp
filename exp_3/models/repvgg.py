import torch
import torch.nn as nn

class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Training branches
        self.rbr_identity = nn.BatchNorm2d(in_channels) if in_channels == out_channels else None
        self.rbr_dense = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.rbr_1x1 = nn.Conv2d(in_channels, out_channels, 1, stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(x)
        
        return self.relu(self.bn(self.rbr_dense(x) + self.rbr_1x1(x) + id_out))


class RepVGG_A0(nn.Module):
    """RepVGG-A0 for CIFAR-10 (simplified)"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Conv2d(3, 48, 3, 1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(48)
        self.relu = nn.ReLU(inplace=True)
        
        # Simplified stages for CIFAR-10
        self.stage1 = self._make_layer(48, 48, 2)
        self.stage2 = self._make_layer(48, 96, 2, stride=2)
        self.stage3 = self._make_layer(96, 192, 2, stride=2)
        self.stage4 = self._make_layer(192, 384, 2, stride=2)
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(384, num_classes)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = [RepVGGBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(RepVGGBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.bn(x)
        x = self.relu(x)
        
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x