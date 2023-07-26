from rembg import remove

def remove_background_rembg(input_path, output_path):
    # 读取图片并进行背景去除
    with open(input_path, "rb") as f:
        image_data = f.read()
        result = remove(image_data)

    # 将去除背景后的图像保存到本地
    with open(output_path, "wb") as f:
        f.write(result)

# 要去除背景的图片路径
input_path = 'path/to/your/image.jpg'  # 替换为你要处理的图片路径

# 处理后的图片保存路径
output_path = 'path/to/output/result.png'  # 替换为你要保存的处理结果图片路径

# 执行图片去背景
remove_background_rembg(input_path, output_path)

print("Background removed and saved to:", output_path)