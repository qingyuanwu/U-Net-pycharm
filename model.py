import torch
import torch.nn as nn  # 通过导入torch.nn，您可以方便地使用这些类和函数来构建和训练神经网络模型，如卷积层，池化层等。
import torch.nn.functional as F  # 通过导入torch.nn.functional，您可以使用这些函数来构建和执行神经网络的各个部分，而不必显式创建层对象，如卷积操作等。
from torch.nn import init   # 通过导入init模块，您可以在构建神经网络模型时使用这些初始化方法，以确保模型参数有一个合适的初始值。

# 定义一个卷积层+ReLU的结合层
class conv_block(nn.Module):
    def __init__(self, channels_input, channels_output):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(                 # nn.Sequential 是 PyTorch 中的一个容器类，通常用于构建神经网络模型的层序列。它允许将多个神经网络层按顺序堆叠在一起，以构建深度神经网络模型。

            nn.Conv2d(channels_input, channels_output, kernel_size=3, stride=1, padding=1, bias=True),  # bias：这是一个布尔值，表示是否在卷积操作中使用偏置项。如果设置为 True，则会为每个输出通道添加一个可学习的偏置项。如果设置为 False，则不会使用偏置项。
            nn.BatchNorm2d(channels_output),   # nn.BatchNorm2d 通常用于卷积神经网络（CNN）的每个卷积层后，以规范每个通道的输出。在实际使用中，您可以将其添加到模型中，以提高训练的稳定性和性能。
            nn.ReLU(inplace=True),    # inplace=True：这是一个可选参数，默认为 False。如果设置为 True，则表示在原地执行激活函数，即将激活函数的输出直接覆盖到输入张量中，而不创建新的张量。这可以节省内存，但会改变原始输入数据。如果设置为 False，则会创建一个新的张量来存储激活函数的输出，而不改变原始输入。
            nn.Conv2d(channels_output, channels_output, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(channels_output),
            nn.ReLU(inplace=True)

        )

    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self, channels_input, channels_output):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(

            nn.Upsample(scale_factor=2),               # 上采样+卷积  效果类似于转置卷积吗？
            nn.Conv2d(channels_input, channels_output, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(channels_output),
            nn.ReLU(inplace=True)

        )

    def forward(self,x):
        x = self.up(x)
        return x

class U_net(nn.Module):
    def __init__(self, img_channels=3, output_channels=1):   # 输入图像为RGB，故为3通道，输出是灰度图，故为1通道。如果输入是灰度图，则img_channels = 1，后面再改为自动判别
        super(U_net, self).__init__()   # 继承父类的构造函数，nn.Module中的？

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)  # 池化层，核大小2*2，步长2

        self.Conv1 = conv_block(channels_input=img_channels, channels_output=64)   # U-net的第一层卷积
        self.Conv2 = conv_block(channels_input=64, channels_output=128)            # U-net的第二层卷积
        self.Conv3 = conv_block(channels_input=128, channels_output=256)           # U-net的第三层卷积
        self.Conv4 = conv_block(channels_input=256, channels_output=512)           # U-net的第四层卷积
        self.Conv5 = conv_block(channels_input=512, channels_output=1024)          # U-net的第五层卷积

        self.Up5 = up_conv(channels_input=1024, channels_output=512)               # U-net的第一次转置卷积
        self.Up_Conv5 = conv_block(channels_input=1024, channels_output=512)       # 第一次转置卷积后维度变为512，但是还要加上第四层的512，故是1024

        self.Up4 = up_conv(channels_input=512, channels_output=256)                # U-net的第二次转置卷积
        self.Up_Conv4 = conv_block(channels_input=512, channels_output=256)

        self.Up3 = up_conv(channels_input=256, channels_output=128)                # U-net的第三次转置卷积
        self.Up_Conv3 = conv_block(channels_input=256, channels_output=128)

        self.Up2 = up_conv(channels_input=128, channels_output=64)                 # U-net的第四次转置卷积
        self.Up_Conv2 = conv_block(channels_input=128, channels_output=64)

        self.Conv_1x1 = nn.Conv2d(64, output_channels, kernel_size=1, stride=1, padding=0)   # 最后有一个1x1的卷积层，论文结构是两层，这里为何是1层？

    def forward(self, x):
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # 转置卷积开始

        d5 = self.Up5(x5)
        # 计算截取区域的起始位置
        h_start = (x4.size(2) - d5.size(2)) // 2
        w_start = (x4.size(3) - d5.size(3)) // 2
        # 使用索引截取子区域
        x4 = x4[:, :, h_start:h_start + d5.size(2), w_start:w_start + d5.size(3)]
        d5 = torch.cat((x4, d5), dim=1)    # 拼接第四层的矢量
        d5 = self.Up_Conv5(d5)

        d4 = self.Up4(d5)
        # 计算截取区域的起始位置
        h_start = (x3.size(2) - d4.size(2)) // 2
        w_start = (x3.size(3) - d4.size(3)) // 2
        # 使用索引截取子区域
        x3 = x3[:, :, h_start:h_start + d4.size(2), w_start:w_start + d4.size(3)]
        d4 = torch.cat((x3, d4), dim=1)    # 拼接第三层的矢量
        d4 = self.Up_Conv4(d4)

        d3 = self.Up3(d4)
        # 计算截取区域的起始位置
        h_start = (x2.size(2) - d3.size(2)) // 2
        w_start = (x2.size(3) - d3.size(3)) // 2
        # 使用索引截取子区域
        x2 = x2[:, :, h_start:h_start + d3.size(2), w_start:w_start + d3.size(3)]
        d3 = torch.cat((x2, d3), dim=1)    # 拼接第二层矢量
        d3 = self.Up_Conv3(d3)

        d2 = self.Up2(d3)
        # 计算截取区域的起始位置
        h_start = (x1.size(2) - d2.size(2)) // 2
        w_start = (x1.size(3) - d2.size(3)) // 2
        # 使用索引截取子区域
        x1 = x1[:, :, h_start:h_start + d2.size(2), w_start:w_start + d2.size(3)]
        d2 = torch.cat((x1, d2), dim=1)     # 拼接第一层矢量
        d2 = self.Up_Conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1