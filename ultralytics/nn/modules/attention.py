
import torch
import torch.nn as nn
import torch.nn.functional as F

# class ECA(nn.Module):
#     def __init__(self, channels, k_size=3):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv = nn.Conv1d(1, 1, kernel_size=k_size, 
#                               padding=k_size//2, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         y = self.avg_pool(x)
#         y = self.conv(y.squeeze(-1).transpose(-1,-2))
#         y = y.transpose(-1,-2).unsqueeze(-1)
#         return x * self.sigmoid(y)

class ECA(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        import math
        # Adaptive kernel size based on channel count
        t = int(abs((math.log2(channels) + b) / gamma))
        k_size = t if t % 2 else t + 1  # ensure odd kernel size
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, 
                              padding=k_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)                          # (B, C, 1, 1)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)) # (B, 1, C)
        y = y.transpose(-1, -2).unsqueeze(-1)          # (B, C, 1, 1)
        return x * self.sigmoid(y)


class SimAM(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w - 1
        mean = x.mean(dim=[2,3], keepdim=True)
        var = ((x - mean) ** 2).sum(dim=[2,3], keepdim=True) / n
        e_inv = (x - mean) ** 2 / (4 * (var + self.e_lambda)) + 0.5
        return x * self.sigmoid(e_inv)
