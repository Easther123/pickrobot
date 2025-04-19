import cv2

def visualize_boxes(image, boxes, labels, scores):
    """
    在图像上绘制检测框和标签
    :param image: 输入图像 (numpy array, H x W x C)
    :param boxes: 检测框坐标 (list of [x1, y1, x2, y2])
    :param labels: 检测类别标签 (list of str)
    :param scores: 检测置信度 (list of float)
    """
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = map(int, box)  # 将坐标转换为整数
        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 添加标签和置信度
        text = f"{label} {score:.2f}"
        cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return image