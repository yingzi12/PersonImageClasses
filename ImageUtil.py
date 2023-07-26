import cv2

def resize_image_with_padding(image, target_size):
    # 获取原始图像尺寸
    height, width = image.shape[:2]

    # 计算缩放比例
    scale = min(target_size[0] / width, target_size[1] / height)

    # 计算缩放后的目标尺寸
    new_width = int(width * scale)
    new_height = int(height * scale)

    # 缩放图像
    resized_image = cv2.resize(image, (new_width, new_height))

    # 创建空白背景画布
    background = 255 * np.ones((target_size[1], target_size[0], 3), dtype=np.uint8)

    # 将缩放后的图像放置在背景中心
    x_offset = (target_size[0] - new_width) // 2
    y_offset = (target_size[1] - new_height) // 2
    background[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image

    return background

# 要缩放的图片路径
image_path = 'path/to/your/image.jpg'  # 替换为你要缩放的图片路径

# 读取图片
image = cv2.imread(image_path)

# 设置目标尺寸（宽度，高度）
target_size = (800, 600)  # 替换为你想要的缩放尺寸

# 缩放图片并进行填充
resized_image = resize_image_with_padding(image, target_size)

# 显示缩放后的结果
cv2.imshow("Resized Image", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()