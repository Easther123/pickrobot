import torch
import torch.nn.functional as F


def multi_modal_similarity(img_feat, text_feat, proto_feat, alpha=0.7):
    """
    融合原型特征和文本特征的相似度计算
    img_feat: Query图像特征 [B, C, H, W]
    text_feat: 文本特征 [1, D]
    proto_feat: 原型特征 [1, C]
    """
    # 原型相似度
    proto_sim = F.cosine_similarity(img_feat, proto_feat, dim=1)

    # 文本相似度（需调整维度）
    text_sim = F.cosine_similarity(img_feat.flatten(2),
                                   text_feat.unsqueeze(-1),
                                   dim=1).mean(dim=2)

    return alpha * proto_sim + (1 - alpha) * text_sim