import os.path

import numpy as np
import torch
from PIL import Image
import csv
from matplotlib import pyplot as plt
from torchvision import transforms as T


model_path = 'D:\Repositories/U-Net/models'
num_epochs = 300
lr = 0.0002
num_epochs_decay = 50
augmentation_prob = 0.4
epoch = 100
epoch_loss = 1
ACC = 1
SE = 2
SP = 3
PC = 4
F1 = 5
JS = 6
DC = 7
result_path = 'D:\Repositories/U-Net/results/'

Norm_ = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # RBG图像，所以是三个通道
GT = Image.open('0.bmp')  # 读入单张图像对应的Ground Truth
plt.imshow(GT)
plt.axis('off')
plt.show()
Transform = []
Transform.append(T.ToTensor())  # 将图像变为张量
Transform = T.Compose(Transform)
GT = Transform(GT)
GT = Norm_(GT)

gray_transform = T.Grayscale(num_output_channels=1)
GT = gray_transform(GT)

plt.imshow(GT[0,:,:])
plt.axis('off')
plt.show()
