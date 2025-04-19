import torchvision
import torch.nn as nn
from torchvision.models import ResNet50_Weights


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, weights=ResNet50_Weights.IMAGENET1K_V1):
        super().__init__()
        resnet = torchvision.models.resnet50(weights=weights)
        self.features = nn.Sequential(*list(resnet.children())[:-2])  # 去掉最后两层（全局池化和全连接层）

    def forward(self, x):
        return self.features(x)  # 输出特征图 [B, 2048, H/32, W/32]