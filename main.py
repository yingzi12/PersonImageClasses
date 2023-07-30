# This is a sample Python script.
import os

import torch
from PIL import Image
import numpy as np
import requests
from ultralytics import YOLO
import cv2
import ImageBackground
import ImageUtil
from Yolo8Vedio import process_video


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# 加载网络图片
def load_image_from_url(url):
    try:
        # 下载图片
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # 将图片数据转换为NumPy数组
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)

        # 从NumPy数组中解码图片
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        return image
    except Exception as e:
        print(f"Error loading image from URL: {str(e)}")
        return None

def yolo8Image(source,outFilePath):

    file_name = os.path.basename(source)
    conf_threshold =0.8
    # Load a pretrained YOLO model (recommended for training)
    model = YOLO('yolov8n.pt')
    results = model(source, conf=conf_threshold)  # list of Results objects
    print('model: %s' % model)
    print("-----------------------------------------")

    i = 0
    j = 0
    for result in results:  # 第二个实例
        i = i + 1
        print('result: %s' % result)
        print("-----------------------------------------")
        boxes = result.boxes
        for box in boxes:  # 第二个实例
            j = j + 1
            print('box: %s' % box)
            print("-----------------------------------------")
            cls = box.cls[0]
            print(cls==0)
            if(cls==0):
                xyxy = box.xyxy[0]
                # 裁剪图片
                # cropped_image = image[int(xyxy[1]) + 1:int(xyxy[3]) + 1, int(xyxy[0]) + 1: int(xyxy[2]) + 1]
                fileName =file_name+ f"{i}-{j}.png";
                # cv2.imwrite(outFileName, cropped_image)
                cutOutFileName=ImageUtil.crop_hd_image(source,outFilePath+"/cutOut/"+fileName,int(xyxy[0]) + 1,int(xyxy[1]) + 1, int(xyxy[2]) + 1, int(xyxy[3]) + 1)
                backFilePath=ImageBackground.remove_background_rembg(fileName,cutOutFileName, outFilePath + "/back");
                ImageUtil.resize_image_with_padding(fileName,backFilePath, outFilePath);
                print("Image saved successfully!")
    model.predict(source, save=True, imgsz=320, conf=0.5, save_crop=True)
def operatorImage(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    # Define remote image or video URL
    source = 'D:\\git\\PersonImageClasses2\\bus.jpg'
    outFilePath = 'D:\\git\\PersonImageClasses2'
    # 加载网络图片
    # loaded_image = load_image_from_url(source)
    if not os.path.exists(outFilePath):
        os.makedirs(outFilePath)
    if not os.path.exists(outFilePath+"/white"):
        os.makedirs(outFilePath+"/white")
    if not os.path.exists(outFilePath+"/cutOut"):
        os.makedirs(outFilePath+"/cutOut")
    if not os.path.exists(outFilePath+"/back"):
        os.makedirs(outFilePath+"/back")
        # 保存高分辨率图像
    if not os.path.exists(outFilePath+"/repair"):
        os.makedirs(outFilePath+"/repair")
    # 加载网络图片
    yolo8Image(source,outFilePath)

def yolo8Vedio():
    folder_path = "E:\\person\\ANYUJIN"  # 替换成实际文件夹路径
    output_folder = 'E:\\person_image\\ANYUJIN'

    # 获取文件夹中的文件和子文件夹列表
    files = os.listdir(folder_path)
    i=0;
    # 输出文件列表
    for file in files:
        # 拼接文件的完整路径
        file_path = os.path.join(folder_path, file)

        # 判断是否为文件
        if os.path.isfile(file_path):
            i = i + 1
            if not os.path.exists(output_folder + "\\" + str(i)):
                os.makedirs(output_folder + "\\" + str(i))
            skip_frames = 1  # 每隔5帧处理一次
            process_video(file_path, output_folder + "\\" + str(i), skip_frames)
            print("File:", file_path)
        # 判断是否为子文件夹
        elif os.path.isdir(file_path):
            print("Folder:", file_path)
        # 可以根据需要继续添加其他类型的判断，例如符号链接等
    # 调用函数示例：
 #   video_file = 'E:\\down\\DM_20230726002609_001.webm'


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(torch.__version__)
    print(torch.cuda.is_available)
    if torch.cuda.is_available():
        print("CUDA is available.")
    else:
        print("CUDA is not available.")
    yolo8Vedio()
    #operatorImage("heihei")
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
