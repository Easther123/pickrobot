# 主要训练步骤
from torch_geometric.graphgym import compute_loss

from utils.similarity import multi_modal_similarity


def train_epoch(support_loader, model, optimizer):
    model.train()
    for support_images, query_images, targets in support_loader:
        # 1. 提取support特征并计算原型
        support_feat = model.extractor(support_images)
        proto_feat = model.prototype_learner(support_feat)

        # 2. 提取query特征
        query_feat = model.extractor(query_images)

        # 3. 计算多模态相似度
        text_feat = model.text_encoder([class_name])
        similarity_map = multi_modal_similarity(query_feat, text_feat, proto_feat)

        # 4. 检测头预测
        cls_pred, reg_pred = model.detection_head(query_feat)

        # 5. 计算损失（分类损失+回归损失+相似度对比损失）
        loss = compute_loss(cls_pred, reg_pred, similarity_map, targets)

        # 6. 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()