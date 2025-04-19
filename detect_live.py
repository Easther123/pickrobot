import os
import cv2
import torch
from torchvision import transforms

from models.detection_head import DetectionHead
from models.resnet_feat import ResNetFeatureExtractor
# 假设其他导入保持不变
# 如果 visualize_boxes 在 utils.visualize 中定义，则需要正确导入它
from utils.visualize import visualize_boxes  # 确保路径和模块名正确

def load_support_features(class_name):
    """
    加载预提取的Support特征
    :param class_name: 类别名称（如 'apple'）
    :return: Support特征 (Tensor, shape: [K, 2048, H, W])
    """
    feature_path = f"data/features/{class_name}.pt"  # 假设特征已保存为 .pt 文件
    return torch.load(feature_path)

def preprocess(image):
    """
    图像预处理
    :param image: 输入图像 (PIL Image 或 numpy array)
    :return: 预处理后的图像 (Tensor, shape: [1, 3, H, W])
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),  # 如果输入是numpy array，需要先转换为PIL Image
        transforms.Resize((224, 224)),  # 调整大小
        transforms.ToTensor(),  # 转换为Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
    ])
    return transform(image).unsqueeze(0)  # 增加 batch 维度

def process_predictions(cls_pred, reg_pred, threshold=0.5):
    """
    处理检测头的输出
    :param cls_pred: 分类预测 (Tensor, shape: [1, 2, H, W])
    :param reg_pred: 回归预测 (Tensor, shape: [1, 4, H, W])
    :param threshold: 置信度阈值
    :return: boxes (list of [x1, y1, x2, y2]), labels (list of str), scores (list of float)
    """
    cls_prob = torch.softmax(cls_pred, dim=1)[0, 1]  # 取目标类别的概率
    cls_prob = cls_prob.squeeze().detach().cpu().numpy()  # 使用 detach() 移除梯度信息后再转换为 numpy array

    print(f"Max confidence score: {cls_prob.max()}")  # 打印最大置信度分数

    mask = cls_prob > threshold
    if not mask.any():
        print("No detections above the threshold.")
        return [], [], []

    boxes = reg_pred[0].permute(1, 2, 0).detach().cpu().numpy()[mask]  # 转换为 [N, 4]
    boxes = boxes * 224  # 假设输入图像大小为 224x224，调整到原图尺寸

    labels = ["apple"] * len(boxes)  # 假设当前类别为 'apple'
    scores = cls_prob[mask].tolist()

    return boxes, labels, scores

def video_detection(class_name, video_path, output_path):
    # 初始化模型
    extractor = ResNetFeatureExtractor().eval()
    detector = DetectionHead(2048)

    # 加载Support Set（假设已经训练好）
    support_feat = load_support_features(class_name)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频文件: {video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 提取Query特征
        query_feat = extractor(preprocess(frame))

        # 检测预测
        cls_pred, reg_pred = detector(query_feat)
        boxes, labels, scores = process_predictions(cls_pred, reg_pred)

        # 可视化检测结果
        output_frame = visualize_boxes(frame, boxes, labels, scores)

        # 写入输出视频
        out.write(output_frame)

        # 显示结果
        cv2.imshow("Fruit Detection", output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 退出
            break

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    class_name = "apple"  # 替换为你要检测的水果类别
    video_path = "data/videos/apple.mp4"  # 输入视频路径
    output_path = "outputs/apple.mp4"  # 输出视频路径

    # 确保 outputs 文件夹存在
    os.makedirs("outputs", exist_ok=True)

    video_detection(class_name, video_path, output_path)