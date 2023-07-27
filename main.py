# This is a sample Python script.
import os

import numpy as np
import requests
from ultralytics import YOLO
import cv2
import ImageBackground
import ImageUtil


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

def yolo8(source,image,outFilePath):
    # Create a new YOLO model from scratch
    model = YOLO('yolov8n.yaml')

    # Load a pretrained YOLO model (recommended for training)
    model = YOLO('yolov8n.pt')
    results = model(source)  # list of Results objects
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
                # cropped_image = image[17:y + height, x:x + width]
                cropped_image = image[int(xyxy[1]) + 1:int(xyxy[3]) + 1, int(xyxy[0]) + 1: int(xyxy[2]) + 1]
                if not os.path.exists(outFilePath):
                    os.makedirs(outFilePath)
                outFileName = f"{outFilePath}\\{i}-{j}.jpg";
                cv2.imwrite(outFileName, cropped_image)
                backFilePath=ImageBackground.remove_background_rembg(outFileName, outFilePath + "\\back");
                ImageUtil.resize_image_with_padding(backFilePath, outFilePath +"\\white");
                print("Image saved successfully!")
    model.predict(source, save=True, imgsz=320, conf=0.5, save_crop=True)
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    # Define remote image or video URL
    source = 'https://ultralytics.com/images/bus.jpg'
    outFilePath = 'C:\\git\\PersonImageClasses2'
    # 加载网络图片
    # loaded_image = load_image_from_url(source)
    # 加载网络图片
    image = load_image_from_url(source)
    yolo8(source,image,outFilePath)
    # if image is not None:
    #     # 显示原始图片
    #     cv2.imshow("Original Image", image)
    #
    #     # 选择裁剪区域
    #     # x, y, width, height = cv2.selectROI("Original Image", image, fromCenter=False, showCrosshair=True)
    #     # print("裁剪:x:%s,y:%s,width:%s,width:%s",x, y, width, height)
    #     # cv2.destroyAllWindows()
    #     #
    #     # xywh: tensor([[409.4020, 499.4990, 784.2324, 537.8137]])
    #     # xywhn: tensor([[0.5054, 0.4625, 0.9682, 0.4980]])
    #     # xyxy: tensor([[17.2858, 230.5922, 801.5182, 768.4059]])
    #     # xyxyn: tensor([[0.0213, 0.2135, 0.9895, 0.7115]])
    #     # 裁剪图片
    #     # cropped_image = image[y:y + height, x:x + width]
    #     cropped_image = image[230:768, 17:801]
    #
    #     # 显示裁剪后的结果
    #     cv2.imshow("Cropped Image", cropped_image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
#    cv2.imwrite("C:\\git\\PersonImageClasses\\data\\cut\\cv_cut_thor.jpg", cropped);
    # Load a model
    # model = YOLO('yolov8n-seg.yaml')  # build a new model from YAML
    # model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)
    # model = YOLO('yolov8n-seg.yaml').load('yolov8n.pt')  # build from YAML and transfer weights



    # Train the model using the 'coco128.yaml' dataset for 3 epochs
    # results = model.train(data='coco128.yaml', epochs=3)

    # Evaluate the model's performance on the validation set
    # results = model.val()



    # Run inference on the source


    #对图片进行截图
    # img = cv2.imread("./data/cut/thor.jpg")
    # print(img.shape)
    # cropped = img[0:128, 0:512]  # 裁剪坐标为[y0:y1, x0:x1]
    # cv2.imwrite("./data/cut/cv_cut_thor.jpg", cropped);

    # Train the model
    # model.train(data='coco128-seg.yaml', epochs=100, imgsz=640)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
