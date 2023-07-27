import os

import cv2
import torch
from PIL import Image
import numpy as np
from RealESRGAN import RealESRGAN
device = torch.device('cpu')

def resize_image_with_padding(fileName,input_path, outFilePath, scale_factor=4):
    # 使用os.path.basename()函数获取文件名称
    file_name = os.path.basename(input_path)
    # 读取图片
    image = cv2.imread(input_path)
    # 设置目标尺寸（宽度，高度）
    target_size = (512, 512)  # 替换为你想要的缩放尺寸
    # 获取原始图像尺寸
    height, width = image.shape[:2]
    if(height<1024):
       input_path= imageRealESRGAN(file_name,input_path,outFilePath+"/repair")
       image = cv2.imread(input_path)
       height, width = image.shape[:2]

    # 计算缩放比例
    scale = min(target_size[0] / width, target_size[1] / height)

    # 计算缩放后的目标尺寸
    new_width = int(width * scale)
    new_height = int(height * scale)

    # 缩放图像
    resized_image = cv2.resize(image, (new_width, new_height))

    # 创建空白背景画布
    high_res_image = 255 * np.ones((target_size[1], target_size[0], 3), dtype=np.uint8)

    # 将缩放后的图像放置在背景中心
    x_offset = (target_size[0] - new_width) // 2
    y_offset = (target_size[1] - new_height) // 2
    high_res_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image

    # 检查目录是否存在，如果不存在则创建目录
    if not os.path.exists(outFilePath):
        os.makedirs(outFilePath)
    whitePath=outFilePath+"/white/white_"+file_name
    cv2.imwrite(whitePath, high_res_image)
    return whitePath

#如果图片不是高清，修复成高清
def imageRealESRGAN(file_name,inputImagePath,outFilePath):
    # 加载Real-ESRGAN模型
    model_path = 'RealESRGAN_x4plus.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = RealESRGAN(device, scale=4)
    model.load_weights('weights/RealESRGAN_x4.pth', download=True)

    path_to_image = inputImagePath
    image = Image.open(path_to_image).convert('RGB')

    sr_image = model.predict(image)


    output_image_path = f"{outFilePath}/{file_name}"
    sr_image.save(output_image_path)
    return output_image_path;

#input_path = 'original_image.png'
# output_path = 'cropped_image.png'
# left = 100  # 裁剪框左上角的x坐标
# upper = 100  # 裁剪框左上角的y坐标
# right = 600  # 裁剪框右下角的x坐标
# lower = 500  # 裁剪框右下角的y坐标
def crop_hd_image(input_path, output_path_name, left, upper, right, lower):
    # 打开原始图像
    image = Image.open(input_path)

    # 裁剪图像
    cropped_image = image.crop((left, upper, right, lower))

    # 保存裁剪后的图像
    cropped_image.save(output_path_name);
    return output_path_name;

#视频转图片
def video_to_images(input_video, output_folder):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error opening video file.")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output_file = f"{output_folder}/frame_{frame_count}.png"
        cv2.imwrite(output_file, frame)
        frame_count += 1

    cap.release()
