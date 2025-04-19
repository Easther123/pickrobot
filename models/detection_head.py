import torch.nn as nn


class DetectionHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 512, 3, padding=1)
        self.cls_head = nn.Conv2d(512, 2, 1)  # 二分类（目标/背景）
        self.reg_head = nn.Conv2d(512, 4, 1)  # 边界框回归

    def forward(self, x):
        x = self.conv(x)
        return self.cls_head(x), self.reg_head(x)