import torch
import torch.nn as nn
import torch.nn.functional as F

def circular_shift(x, dx, dy):
    """Circular shift along spatial dimensions"""
    if dx == 0 and dy == 0:
        return x
    return torch.roll(x, shifts=(dx, dy), dims=(2, 3))

class ShiftwiseConv(nn.Module):
    """Shiftwise Convolution layer"""
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 paths=8, stride=1, padding=1, bias=False):
        super().__init__()
        self.paths = paths
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding
        
        # Generate shift offsets (for 7x7 equivalent receptive field)
        self.shifts = self._generate_shifts(paths)
        
        # Each path has its own 3x3 convolution
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size, 
                      stride=1, padding=padding, bias=bias)
            for _ in range(paths)
        ])
        
        # Batch norm after fusion
        self.bn = nn.BatchNorm2d(out_channels)
        
    def _generate_shifts(self, paths):
        """Generate shift offsets for emulating different kernel sizes"""
        shifts = []
        
        # For 7x7 equivalent: 8 directions (3x3 grid without center)
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                shifts.append((dx, dy))
        
        # If we need more paths, add larger shifts
        if len(shifts) < paths:
            # Add shifts with larger offsets for bigger receptive field
            for offset in [2, 3]:
                if len(shifts) >= paths:
                    break
                for dy in [-offset, offset]:
                    if len(shifts) < paths:
                        shifts.append((0, dy))
                    if len(shifts) < paths:
                        shifts.append((dy, 0))
        
        # Trim to requested number of paths
        return shifts[:paths]
    
    def forward(self, x):
        outputs = []
        for i, conv in enumerate(self.convs):
            dx, dy = self.shifts[i]
            shifted = circular_shift(x, dx, dy)
            out = conv(shifted)
            outputs.append(out)
        
        # Sum all paths
        out = torch.stack(outputs, dim=0).sum(dim=0)
        
        # Apply stride via average pooling if needed
        if self.stride > 1:
            out = F.avg_pool2d(out, self.stride)
        
        out = self.bn(out)
        return out