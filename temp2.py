import os.path
from PIL import Image

GT = Image.open('D:\Repositories/U-Net/train_GT/0.bmp')
channels = GT.mode
print("该图像的颜色模式为:", channels)