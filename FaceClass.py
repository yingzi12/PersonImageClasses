import os
import cv2
import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from sklearn.svm import SVC
import numpy as np

# 检查CUDA是否可用，如果可以则使用CUDA进行计算，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 第1步：加载预训练的VGG-Face模型
def load_vgg_face_model():
    model = timm.create_model("vgg16", pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])
    model = model.to(device)
    model.eval()
    return model

# 第2步：加载SVM分类器
classifier = SVC(kernel='linear', C=1.0)

# 第3步：定义一个函数来从图像中提取特征
def extract_features(image_path, model):
    try:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将BGR转换为RGB
        img = cv2.resize(img, (224, 224))  # 调整大小为VGG-Face输入大小

        # 将图像转换为张量并移动到GPU（如果可用）
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img_tensor = transform(Image.fromarray(img)).unsqueeze(0).to(device)

        # 使用VGG-Face模型提取特征
        with torch.no_grad():
            features = model(img_tensor).squeeze().cpu().numpy()

        return features
    except Exception as e:
        print(f"从图像提取特征时出现错误：{e}")
        return None

# 第4步：加载数据并训练分类器
def load_data_and_train_classifier(train_folder, model):
    features_list = []
    labels_list = []

    for label, subfolder in enumerate(os.listdir(train_folder)):
        subfolder_path = os.path.join(train_folder, subfolder)
        if os.path.isdir(subfolder_path):
            for file in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, file)
                features = extract_features(file_path, model)
                if features is not None:
                    features_list.append(features)
                    labels_list.append(label)

    features = np.array(features_list)
    labels = np.array(labels_list)

    classifier.fit(features, labels)

# 第5步：加载测试数据并对图像进行分类
def classify_images(test_folder, model):
    for file in os.listdir(test_folder):
        file_path = os.path.join(test_folder, file)
        features = extract_features(file_path, model)
        if features is not None:
            label = classifier.predict([features])
            print(f"图像'{file}'属于类别 {label}.")

if __name__ == "__main__":
    # 用你自己的数据文件夹替换这些路径
    train_folder = "E:\\person_tain"
    test_folder = "E:\\person_image\\ANYUJIN\\2"

    # 第6步：加载VGG-Face模型
    model = load_vgg_face_model()

    # 第7步：加载数据并训练分类器
    load_data_and_train_classifier(train_folder, model)

    # 第8步：对测试文件夹中的图像进行分类
    classify_images(test_folder, model)
