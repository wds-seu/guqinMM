import os
from PIL import Image
from torchvision import transforms

def convert_images_to_grayscale(input_folder, output_folder):
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith((".png", ".jpg", ".jpeg", ".bmp")):  # 只处理图像文件
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # 打开图像
            image = Image.open(input_path)

            # 将图像转换为灰度图像
            image_gray = image.convert("L")

            # 保存灰度图像到输出文件夹
            image_gray.save(output_path)
            print(f"Converted {filename} to grayscale and saved to {output_path}")

# 示例用法
input_folder = "./data/wushen_jzp/Music/images"
output_folder = "./data/wushen_jzp/images_gray"
convert_images_to_grayscale(input_folder, output_folder)