import torch
import os
from torchvision import transforms
from models.resnet_feat import ResNetFeatureExtractor
from data_loader.support_dataset import SupportSet


def extract_support_features(class_name, support_dir, output_dir):
    """
    提取 Support 图像的特征并保存为 .pt 文件
    :param class_name: 类别名称（如 'apple'）
    :param support_dir: Support 图像目录
    :param output_dir: 特征保存目录
    """
    # 初始化模型
    extractor = ResNetFeatureExtractor().eval()

    # 定义图像预处理流程
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整大小
        transforms.ToTensor(),  # 转换为Tensor
        transforms.Normalize(  # 归一化
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # 加载 Support 数据集
    support_set = SupportSet(root_dir=support_dir, class_name=class_name)

    if len(support_set) == 0:
        print(f"警告: 在 {os.path.join(support_dir, class_name)} 中没有找到任何图片。")
        return

    # 提取特征
    features = []
    for idx, image in enumerate(support_set):
        image = preprocess(image)  # 预处理图像
        with torch.no_grad():
            feat = extractor(image.unsqueeze(0))  # 增加 batch 维度
            features.append(feat.squeeze(0))  # 去掉 batch 维度
        print(f"已处理第 {idx + 1} 张图片")

    # 将特征保存为 .pt 文件
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{class_name}.pt")
    torch.save(features, output_path)
    print(f"特征已保存到 {output_path}")


if __name__ == "__main__":
    class_name = "apple"  # 替换为你的类别
    base_dir = "C:\\Users\\29901\\PycharmProjects\\Pick-Robot"  # 使用绝对路径避免路径拼接问题
    support_dir = os.path.join(base_dir, "data", "support")  # 使用 os.path 处理路径
    output_dir = os.path.join(base_dir, "data", "features")

    # 检查路径是否存在
    class_path = os.path.join(support_dir, class_name)
    if not os.path.exists(class_path):
        raise FileNotFoundError(f"目录 {class_path} 不存在！请检查 Support 图像是否放置正确。")

    print(f"开始处理 {class_name} 的 Support 图像...")
    extract_support_features(class_name, support_dir, output_dir)