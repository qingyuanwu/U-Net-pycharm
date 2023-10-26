import torch
from PIL import Image
import cv2

# 1. 打开图像文件
image = cv2.imread("0.bmp", cv2.IMREAD_COLOR)

# 2. 将图像转换为PyTorch张量
image_tensor = torch.tensor(image)

channels = 1
# 提取image_tensor的长度与宽度
width, height = image_tensor.shape[0], image_tensor.shape[1]
# 使用 view 方法来重塑张量
new_shape = (-1, width, height, channels)
reshaped_tensor = image_tensor.view(new_shape)   # 疑问：为什么要这样设置维度，四维？