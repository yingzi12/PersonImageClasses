#用ResNet50模型提取人物照片的特征向量。然后，
# 使用K-Means算法对特征向量进行聚类，将相似的特征向量聚在一起形成不同的类别
# 。输出结果将显示每个类别中的人物照片路径。
import cv2
import numpy as np
from keras.applications import ResNet50, preprocess_input
from sklearn.cluster import KMeans

def extract_features(image, model):
    # 图像预处理
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    # 提取图像特征向量
    features = model.predict(image)
    return features.flatten()

# 加载ResNet50模型
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# 要分类的人物照片路径列表
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', 'path/to/image3.jpg', ...]

# 提取图像特征向量
features_list = []
for path in image_paths:
    image = cv2.imread(path)
    features = extract_features(image, resnet_model)
    features_list.append(features)

# 使用K-Means进行相似度聚类
num_clusters = 5  # 替换为聚类的簇数
kmeans = KMeans(n_clusters=num_clusters)
labels = kmeans.fit_predict(features_list)

# 打印分类结果
for i in range(num_clusters):
    print(f"Cluster {i+1}:")
    for j, label in enumerate(labels):
        if label == i:
            print(image_paths[j])