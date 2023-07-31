#要根据文件的相似度来去重，可以使用图像特征向量的余弦相似度来衡量图像之间的相似度。
# 我们可以使用ResNet50模型提取图像的特征向量，然后计算特征向量之间的余弦相似度。
# 如果余弦相似度超过某个阈值，则认为这两个图像是相似的，并将其中一个视为重复图像。
import os
import shutil

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

# 如果可用，加载CUDA设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_features(image_path, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path)
    img_tensor = transform(img)
    img_tensor = torch.unsqueeze(img_tensor, 0).to(device)
    with torch.no_grad():
        features = model(img_tensor)
    features = features.squeeze().cpu().numpy()
    return features

def cosine_similarity(features1, features2):
    dot_product = np.dot(features1, features2)
    norm1 = np.linalg.norm(features1)
    norm2 = np.linalg.norm(features2)
    similarity = dot_product / (norm1 * norm2)
    return similarity

def remove_duplicates(input_folder, output_folder, threshold=0.9):
    # 如果输出文件夹不存在，则创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 加载带有CUDA的ResNet50模型
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    model.to(device)
    model.eval()

    # 遍历输入文件夹中的所有文件和子文件夹
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, file)
                output_subfolder = os.path.relpath(root, input_folder)
                output_subfolder_path = os.path.join(output_folder, output_subfolder)
                # 处理当前图像文件
                process_image(image_path, model, output_subfolder_path, threshold)

    print("重复照片去除完成。")

def process_image(image_path, model, output_folder, threshold):
    # 如果输出子文件夹不存在，则创建输出子文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 提取图像的特征向量
    features1 = extract_features(image_path, model)

    is_duplicate = False
    # 遍历输出子文件夹中的已处理过的图像文件
    for processed_image in os.listdir(output_folder):
        processed_image_path = os.path.join(output_folder, processed_image)

        # 提取已处理图像的特征向量
        features2 = extract_features(processed_image_path, model)

        # 计算特征向量的余弦相似度
        similarity = cosine_similarity(features1, features2)

        # 如果相似度超过阈值，则认为是重复图像
        if similarity > threshold:
            is_duplicate = True
            break

    if not is_duplicate:
        # 将非重复图像复制到输出子文件夹
        shutil.copy(image_path, os.path.join(output_folder, os.path.basename(image_path)))
        print(f"唯一照片：{os.path.basename(image_path)}")


if __name__ == "__main__":
    input_folder = "E:\\person_image_copy\\ANYUJIN"  # 替换为包含要去除重复的图像的文件夹路径
    output_folder = "E:\\person_image_copy_2\\ANYUJIN"  # 替换为存储去除重复图像的文件夹路径
    remove_duplicates(input_folder, output_folder)


