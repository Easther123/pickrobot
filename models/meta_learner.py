import torch
import torch.nn as nn

class PrototypeLearner(nn.Module):
    def compute_prototypes(self, support_features):
        """support_features shape: [K, 2048, h, w]"""
        return torch.mean(support_features, dim=0)  # 原型特征聚合