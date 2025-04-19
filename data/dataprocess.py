import os
import shutil

# 定义源路径和目标路径
source_paths = [
    "C:\\Users\\29901\\Desktop\\classData\\test",
    "C:\\Users\\29901\\Desktop\\classData\\val"
]
target_path_support = "C:\\Users\\29901\\PycharmProjects\\Pick-Robot\\data\\support"
target_path_query = "C:\\Users\\29901\\PycharmProjects\\Pick-Robot\\data\\query"

# 英文名称映射
fruit_names = {
    '哈密瓜': 'honeydew',
    '杏': 'apricot',
    '柠檬': 'lemon',
    '柿子': 'persimmon',
    '桃子': 'peach',
    '梨': 'pear',
    '樱桃': 'cherry',
    '橙子': 'orange',
    '百香果': 'passionfruit',
    '红萝卜': 'carrot',
    '苹果': 'apple',
    '草莓': 'strawberry',
    '荔枝': 'lychee',
    '葡萄': 'grape',
    '西瓜': 'watermelon',
    '香蕉': 'banana',
    '西红柿': 'tomato'
}

# 处理每个源路径
for source_path in source_paths:
    print(f"正在处理源路径: {source_path}")
    for folder_name in os.listdir(source_path):
        folder_path = os.path.join(source_path, folder_name)
        if os.path.isdir(folder_path):
            # 查找images文件夹
            images_folder_path = os.path.join(folder_path, 'images')
            if not os.path.exists(images_folder_path):
                print(f"  警告: 文件夹 '{folder_name}' 下没有 'images' 子文件夹，将被跳过。")
                continue

            # 获取文件夹内的所有图片（包括不同格式）
            images = [f for f in os.listdir(images_folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            # 确定目标路径
            target_folder = target_path_support if 'test' in source_path else target_path_query

            if folder_name not in fruit_names:
                print(f"    警告: 文件夹 '{folder_name}' 的名称未在映射字典中找到，将被跳过。")
                continue

            # 创建目标文件夹（如果不存在）
            dest_dir = os.path.join(target_folder, fruit_names[folder_name])
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)

            # 移动并重命名图片
            for i, image in enumerate(images):
                src = os.path.join(images_folder_path, image)
                dst = os.path.join(dest_dir, f"{fruit_names[folder_name]}{i + 1}.jpg")
                print(f"    将文件从 {src} 复制到 {dst}")
                try:
                    shutil.copy(src, dst)  # 使用copy代替move以保留原始文件
                except Exception as e:
                    print(f"    错误: 无法复制文件 {src} 到 {dst}, 错误原因: {e}")

print("文件组织完成！")