'''

这段代码定义了一个自定义的 PyTorch 数据集类 ImageFolder，用于加载图像数据并进行预处理。这个类继承自 torch.utils.data.Dataset 类，需要实现 __init__、__getitem__ 和 __len__ 方法，以便可以被 PyTorch 的数据加载器使用。

以下是该类的主要功能：

__init__(self, root, image_size=224, mode='train', augmentation_prob=0.4)：初始化方法，用于设置数据集的根目录、图像大小、模式（训练或其他）、数据增强概率等参数。

__getitem__(self, index)：根据给定索引加载图像和其对应的地面真相（Ground Truth），并应用数据增强。这包括对图像进行随机旋转、裁剪、翻转、亮度、对比度调整等操作，以及规范化。

__len__(self)：返回数据集的长度，即包含的图像数量。

这个类的主要用途是用于训练深度学习模型时的数据加载和预处理。它可以帮助您管理图像数据集，对图像进行增强和规范化，以便用于训练神经网络模型。此类通常与 PyTorch 的数据加载器一起使用，以批量加载和处理数据。

'''

import os
import random
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image

class ImageFolder(data.Dataset):
    def __init__(self, root, mode='train', augmentation_prob=0.4):    # 这是类的构造方法，用于初始化对象的属性和状态。
        self.root = root                 # 训练集的待处理图像地址，如D:\Repositories/U-Net/train/

        self.GT_root = root[:-1] + '_GT/'     # 训练集的ground truth保存在D:\Repositories/U-Net/train_GT/
        self.img_path = list(map(lambda x: os.path.join(root, x), os.listdir(root)))   # 返回一个root下的所有子文件的列表，并将其和root进行连接，连接后img_path的每一个元素就是一张图片的地址
        self.mode = mode     # 这样写有什么优势？
        self.Rotation_Degree = [0, 90, 180, 270]    # 用来旋转图像，增加训练集数量
        self.augmentation_prob = augmentation_prob       # 用来一个参数，用来判断是否要进行训练集旋转等操作

        print("Image count in {} path: {}".format(self.mode, len(self.img_path)))

    def __getitem__(self, index):        # 这是一个用于获取数据集中单个数据样本的方法。

        image_path = self.img_path[index]
        file_name = image_path.split('/')[-1][:-len(".bmp")]    # 将image_path中的string，按照’/‘进行分块，只读取最后一块，并省略后缀’.bmp‘，这样file_name就是一张图片的名字
        GT_path = self.GT_root + file_name + '.bmp'

        image = Image.open(image_path)  # 读入单张图像
        GT = Image.open(GT_path)        # 读入单张图像对应的Ground Truth

        aspect_ratio = image.size[1]/image.size[0]    # 下面进行图像增强操作，即变换图像大小，旋转等操作。计算长宽比

        Transform = []                  # 用来保存图像变换的方法，用T.Compose(Transform)将所有的操作按顺序合成一个

        Resize_Range = random.randint(500,520)        # 设定图像放大缩小的范围，本次训练的图像尺寸为512x512
        Transform.append(T.Resize((int(Resize_Range*aspect_ratio), Resize_Range)))  # 变换图像大小，并将这个操作加在Transform最后

        transform_prob = random.random()     # 设置旋转等操作的发生概率

        if (self.mode == 'train') and transform_prob <= self.augmentation_prob:        # 如果是训练模式，且数据增强概率小于augmentation
            Rotation_Degree = random.randint(0, 3)                         # 设置旋转操作
            Rotation_Degree = self.Rotation_Degree[Rotation_Degree]
            if (Rotation_Degree == 90) or (Rotation_Degree == 270):        # 因为旋转90和270时，图像的长宽比变化了，所以要改变
                aspect_ratio = 1/aspect_ratio
            Transform.append(T.RandomRotation((Rotation_Degree, Rotation_Degree)))   # T.RandomRotation的第一个参数()，中设置随机旋转角度的上限与下限

            Rotation_Range = random.randint(-10, 10)                       # 其他的旋转操作
            Transform.append(T.RandomRotation((Rotation_Range, Rotation_Range)))

            Crop_Range = random.randint(450, 470)
            Transform.append(T.CenterCrop((int(Crop_Range*aspect_ratio), Crop_Range)))  # 中心裁剪尺寸

            Transform = T.Compose(Transform)          # 将上面的图像操作打包到Transform中

            image = Transform(image)
            GT = Transform(GT)

            Shift_Range_Left = random.randint(0, 20)             # 设置裁剪坐标
            Shift_Range_Upper = random.randint(0,20)
            Shift_Range_Right = image.size[1]-random.randint(0, 20)
            Shift_Range_lower = image.size[0]-random.randint(0, 20)
            image = image.crop(box=(Shift_Range_Left, Shift_Range_Upper, Shift_Range_Right, Shift_Range_lower))
            GT = GT.crop(box=(Shift_Range_Left, Shift_Range_Upper, Shift_Range_Right, Shift_Range_lower))

            if random.random() < 0.5:
                image = F.hflip(image)
                GT = F.hflip(GT)

            if random.random() < 0.5:
                image = F.vflip(image)
                GT = F.vflip(GT)

            Transform = T.ColorJitter(brightness=0.2, contrast=0.2, hue=0.02)

            image = Transform(image)

            Transform = []

        Transform.append(T.ToTensor())       # 将图像变为张量
        Transform = T.Compose(Transform)

        image = Transform(image)
        GT = Transform(GT)

        Norm_ = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))     # RBG图像，所以是三个通道
        image = Norm_(image)

        return image, GT

    def __len__(self):                   # 这个方法用于返回数据集的长度，通常是数据样本的总数。

        return len(self.img_path)

def get_loader(image_path, batch_size, num_workers=8, mode='train', augmentation_prob=0.4):    # 若mode='train'，则在ImageFolder中要进行图像增强

    dataset = ImageFolder(root=image_path, mode=mode, augmentation_prob=augmentation_prob)    # 图像被预处理及数据增强结束

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  sampler=None,
                                  num_workers=num_workers,
                                  )
    return data_loader
