import os
import cv2
import numpy as np
from ultralytics import YOLO

# 加载预训练的 YOLO 模型
model = YOLO('yolov5s.pt')

def process_frame(frame, frame_count, output_folder, skip_frames=5):
    # 在每 `skip_frames` 帧上运行处理
    if frame_count % skip_frames == 0:
        # 使用 YOLO 模型检测物体
        results = model(frame)

        for result in results:
            for box in result.boxes:
                cls = box.cls[0]
                # 如果检测到物体类别是你感兴趣的，就保存图像帧
                if cls == 0:  # 例如，这里的 cls==0 表示检测到的类别为第一类
                    xyxy = box.xyxy[0]
                    detected_frame = frame[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                    output_file = os.path.join(output_folder, f"frame_{frame_count}_object.png")
                    cv2.imwrite(output_file, detected_frame)

def process_video(input_video, output_folder, skip_frames=5):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error opening video file.")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 处理视频帧
        process_frame(frame, frame_count, output_folder, skip_frames)

        frame_count += 1

    cap.release()

# 调用函数示例：
video_file = 'input_video.mp4'
output_folder = 'output_detected_images'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
skip_frames = 5  # 每隔5帧处理一次
process_video(video_file, output_folder, skip_frames)
