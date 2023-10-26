from __future__ import division
import numpy as np
import pandas as pd
import cv2

'''
数据集预处理
'''
# 读取数据集（CSV文件中的mask和训练数据）
# 此时的数据结构为DataFrame，每个单元的内容是图片的地址
csv_mask_data = pd.read_csv('D:\Repositories/U-Net/GlandsMask.csv')
csv_image_data = pd.read_csv('D:\Repositories/U-Net/GlandsImage.csv')

# 从csv_mask_data和csv_image_data中选择所有的行和所有的列，
# 然后将它们作为一个NumPy数组存储在mask_data和image_data变量中。
# 此时的数据结构为ndarray，每个单元的内容是图片的地址
mask_data = csv_mask_data.iloc[:, :].values
image_data = csv_image_data.iloc[:, :].values

# 将mask_data和image_data的内容顺序打乱，但是它们仍然是相互对应的
perm = np.arange(len(csv_mask_data))
np.random.shuffle(perm)
mask_data = mask_data[perm]
image_data = image_data[perm]
