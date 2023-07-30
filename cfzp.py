#去重复照片
import os
import shutil
import torch
import torchvision.transforms as transforms
from PIL import Image
import imagehash


def get_perceptual_hash(image_path):
    # 加载图像并将其转换为灰度图像
    img = Image.open(image_path).convert('L')

    # 将图像调整为固定尺寸以进行一致的哈希计算
    img = img.resize((8, 8), Image.ANTIALIAS)

    # 计算图像的感知哈希
    return imagehash.phash(img)


def remove_duplicates(input_folder, output_folder):
    # 如果输出文件夹不存在，则创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 列出输入文件夹中的所有图像文件
    image_files = [file for file in os.listdir(input_folder) if file.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # 初始化一个字典来存储感知哈希和对应的文件名
    hash_dict = {}

    # 如果可用，加载CUDA设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 定义转换以对ResNet50中的图像进行归一化处理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载带有CUDA的ResNet50模型
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    model.to(device)
    model.eval()

    # 遍历所有图像文件
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)

        # 计算图像的感知哈希
        p_hash = get_perceptual_hash(image_path)

        # 检查感知哈希是否已在哈希字典中
        if p_hash in hash_dict:
            # 如果哈希已经在字典中，则说明是重复图像
            print(f"重复照片：{image_file}")
        else:
            # 如果哈希不在字典中，则说明是新图像
            hash_dict[p_hash] = image_file

            # 将新图像复制到输出文件夹
            shutil.copy(image_path, os.path.join(output_folder, image_file))
            print(f"唯一照片：{image_file}")

    print("重复照片去除完成。")


if __name__ == "__main__":
    input_folder = "替换为你的输入文件夹路径"  # 替换为包含要去除重复的图像的文件夹路径
    output_folder = "替换为你的输出文件夹路径"  # 替换为存储去除重复图像的文件夹路径
    remove_duplicates(input_folder, output_folder)
