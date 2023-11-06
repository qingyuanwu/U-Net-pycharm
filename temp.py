import os.path
import random
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from model import U_net
from torchvision.transforms import functional as F
from torch import optim
import argparse
from torch.backends import cudnn
import torch.nn.functional
from evaluation import *
import csv

class ImageFolder(data.Dataset):
    def __init__(self, root, image_size=160, mode='train', augmentation_prob=0.4):    # 这是类的构造方法，用于初始化对象的属性和状态。
        self.root = root                 # 训练集的待处理图像地址，如D:\Repositories/U-Net/train/

        self.GT_root = root[:-1] + '_GT/'     # 训练集的ground truth保存在D:\Repositories/U-Net/train_GT/
        self.img_path = list(map(lambda x: os.path.join(root, x), os.listdir(root)))   # 返回一个root下的所有子文件的列表，并将其和root进行连接，连接后img_path的每一个元素就是一张图片的地址
        self.img_size = image_size
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

        # aspect_ratio = image.size[1] / image.size[0]  # 下面进行图像增强操作，即变换图像大小，旋转等操作。计算长宽比
        #
        # Transform = []  # 用来保存图像变换的方法，用T.Compose(Transform)将所有的操作按顺序合成一个
        #
        # Rotation_Degree = random.randint(0, 3)  # 设置旋转操作
        # Rotation_Degree = self.Rotation_Degree[Rotation_Degree]
        # if (Rotation_Degree == 90) or (Rotation_Degree == 270):  # 因为旋转90和270时，图像的长宽比变化了，所以要改变
        #     aspect_ratio = 1 / aspect_ratio
        # Transform.append(
        #     T.RandomRotation((Rotation_Degree, Rotation_Degree)))  # T.RandomRotation的第一个参数()，中设置随机旋转角度的上限与下限
        #
        # Rotation_Range = random.randint(-10, 10)  # 其他的旋转操作
        # Transform.append(T.RandomRotation((Rotation_Range, Rotation_Range)))
        #
        # Resize_Range = random.randint(500, 520)  # 设定图像放大缩小的范围，本次训练的图像尺寸为512x512
        # Transform.append(T.Resize((int(Resize_Range * aspect_ratio), Resize_Range)))  # 变换图像大小，并将这个操作加在Transform最后
        # Transform.append(T.ToTensor())
        #
        # Transform = T.Compose(Transform)  # 将上面的图像操作打包到Transform中
        #
        # image = Transform(image)
        # GT = Transform(GT)

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

        # Transform.append(T.Resize(((int(256*aspect_ratio)-int(256*aspect_ratio))%16, 256)))    # 进一步扭曲图像，增加训练数目
        Transform.append(T.ToTensor())       # 将图像变为张量
        Transform = T.Compose(Transform)

        image = Transform(image)
        GT = Transform(GT)

        Norm_ = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))     # RBG图像，所以是三个通道
        image = Norm_(image)

        return image, GT

    def __len__(self):                   # 这个方法用于返回数据集的长度，通常是数据样本的总数。

        return len(self.img_path)

def custom_collate(batch):
    if len(batch) != 2:
        raise ValueError("Each batch should contain exactly 2 elements: image and GT.")
    image, GT = batch
    return image, GT

class Solver(object):
    def __init__(self, config, train_loader):

        self.train_loader = train_loader

        self.collate_fn = custom_collate

        self.unet = None
        self.optimizer = None
        self.image_channels = config.image_channels
        self.output_channels = config.output_channels
        self.criterion = torch.nn.BCELoss()               # 损失函数
        self.augmentation_prob = config.augmentation_prob

        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        self.num_epochs = config.num_epoch
        self.num_epochs_decay = config.num_epoch_decay
        self.batch_size = config.batch_size

        self.log_step = config.log_step
        self.val_step = config.val_step

        self.model_path = config.model_path
        self.result_path = config.result_path

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.build_model()

    def build_model(self):
        self.unet = U_net(img_channels=3, output_channels=1)
        self.optimizer = optim.Adam(list(self.unet.parameters()), self.lr, [self.beta1, self.beta2])
        self.unet.to(self.device)


    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))


    def to_data(self, x):
        """Convert variable to tensor."""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data

    def reset_grad(self):
        """Zero the gradient buffers."""
        self.unet.zero_grad()


    def compute_accuracy(self, SR, GT):
        SR_flat = SR.view(-1)
        GT_flat = GT.view(-1)

        acc = GT_flat.data.cpu() == (SR_flat.data.cpu() > 0.5)


    def tensor2img(self, x):
        img = (x[:, 0, :, :] > x[:, 1, :, :]).float()
        img = img * 255
        return img

    def train(self):
        """Train encoder, generator and discriminator."""

        # ====================================== Training ===========================================#
        # ===========================================================================================#

        unet_path = os.path.join(self.model_path, 'U_net-%d-%.4f-%d-%.4f.pkl' % (
        self.num_epochs, self.lr, self.num_epochs_decay, self.augmentation_prob))

        # U-Net Train
        if os.path.isfile(unet_path):
            # Load the pretrained Encoder
            self.unet.load_state_dict(torch.load(unet_path))
            print('U_net is Successfully Loaded from %s' % (unet_path))
        else:
            # Train for Encoder
            lr = self.lr
            best_unet_score = 0

            for epoch in range(self.num_epochs):

                self.unet.train(True)
                epoch_loss = 0

                acc = 0.  # Accuracy
                SE = 0.  # Sensitivity (Recall)
                SP = 0.  # Specificity
                PC = 0.  # Precision
                F1 = 0.  # F1 Score
                JS = 0.  # Jaccard Similarity
                DC = 0.  # Dice Coefficient
                length = 0

                for i, (images, GT) in enumerate(self.train_loader):
                    # GT : Ground Truth

                    images = images.to(self.device)
                    GT = GT.to(self.device)
                    gray_transform = T.Grayscale(num_output_channels=1)
                    GT = gray_transform(GT)

                    # SR : Segmentation Result
                    SR = self.unet(images)
                    SR_probs = torch.nn.functional.sigmoid(SR)
                    SR_flat = SR_probs.view(SR_probs.size(0), -1)

                    width, height = SR.size(2), SR.size(3)
                    GT = torch.nn.functional.interpolate(GT, size=(height, width), mode='bilinear', align_corners=False)
                    GT_flat = GT.view(GT.size(0), -1)
                    loss = self.criterion(SR_flat, GT_flat)
                    epoch_loss += loss.item()

                    # Backprop + optimize
                    self.reset_grad()
                    loss.backward()
                    self.optimizer.step()

                    acc += get_accuracy(SR, GT)
                    SE += get_sensitivity(SR, GT)
                    SP += get_specificity(SR, GT)
                    PC += get_precision(SR, GT)
                    F1 += get_F1(SR, GT)
                    JS += get_JS(SR, GT)
                    DC += get_DC(SR, GT)
                    length += images.size(0)

                acc = acc / length
                SE = SE / length
                SP = SP / length
                PC = PC / length
                F1 = F1 / length
                JS = JS / length
                DC = DC / length

                # Print the log info
                print(
                    'Epoch [%d/%d], Loss: %.4f, \n[Training] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f' % (
                        epoch + 1, self.num_epochs, \
                        epoch_loss, \
                        acc, SE, SP, PC, F1, JS, DC))

                # Decay learning rate
                if (epoch + 1) > (self.num_epochs - self.num_epochs_decay):
                    lr -= (self.lr / float(self.num_epochs_decay))
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr
                    print('Decay learning rate to lr: {}.'.format(lr))

                # ===================================== Validation ====================================#
                self.unet.train(False)
                self.unet.eval()

                acc = 0.  # Accuracy
                SE = 0.  # Sensitivity (Recall)
                SP = 0.  # Specificity
                PC = 0.  # Precision
                F1 = 0.  # F1 Score
                JS = 0.  # Jaccard Similarity
                DC = 0.  # Dice Coefficient
                length = 0
                for i, (images, GT) in enumerate(self.valid_loader):
                    images = images.to(self.device)
                    GT = GT.to(self.device)
                    SR = torch.nn.functional.sigmoid(self.unet(images))
                    acc += get_accuracy(SR, GT)
                    SE += get_sensitivity(SR, GT)
                    SP += get_specificity(SR, GT)
                    PC += get_precision(SR, GT)
                    F1 += get_F1(SR, GT)
                    JS += get_JS(SR, GT)
                    DC += get_DC(SR, GT)

                    length += images.size(0)

                acc = acc / length
                SE = SE / length
                SP = SP / length
                PC = PC / length
                F1 = F1 / length
                JS = JS / length
                DC = DC / length
                unet_score = JS + DC

                print('[Validation] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f' % (
                acc, SE, SP, PC, F1, JS, DC))

                '''
                torchvision.utils.save_image(images.data.cpu(),
                                            os.path.join(self.result_path,
                                                        '%s_valid_%d_image.png'%(self.model_type,epoch+1)))
                torchvision.utils.save_image(SR.data.cpu(),
                                            os.path.join(self.result_path,
                                                        '%s_valid_%d_SR.png'%(self.model_type,epoch+1)))
                torchvision.utils.save_image(GT.data.cpu(),
                                            os.path.join(self.result_path,
                                                        '%s_valid_%d_GT.png'%(self.model_type,epoch+1)))
                '''

                # Save Best U-Net model
                if unet_score > best_unet_score:
                    best_unet_score = unet_score
                    best_epoch = epoch
                    best_unet = self.unet.state_dict()
                    print('Best U_net model score : %.4f' % (best_unet_score))
                    torch.save(best_unet, unet_path)

            # ===================================== Test ====================================#
            del self.unet
            del best_unet
            self.build_model()
            self.unet.load_state_dict(torch.load(unet_path))

            self.unet.train(False)
            self.unet.eval()

            acc = 0.  # Accuracy
            SE = 0.  # Sensitivity (Recall)
            SP = 0.  # Specificity
            PC = 0.  # Precision
            F1 = 0.  # F1 Score
            JS = 0.  # Jaccard Similarity
            DC = 0.  # Dice Coefficient
            length = 0
            for i, (images, GT) in enumerate(self.valid_loader):
                images = images.to(self.device)
                GT = GT.to(self.device)
                SR = torch.nn.functional.sigmoid(self.unet(images))
                acc += get_accuracy(SR, GT)
                SE += get_sensitivity(SR, GT)
                SP += get_specificity(SR, GT)
                PC += get_precision(SR, GT)
                F1 += get_F1(SR, GT)
                JS += get_JS(SR, GT)
                DC += get_DC(SR, GT)

                length += images.size(0)

            acc = acc / length
            SE = SE / length
            SP = SP / length
            PC = PC / length
            F1 = F1 / length
            JS = JS / length
            DC = DC / length
            unet_score = JS + DC

            f = open(os.path.join(self.result_path, 'result.csv'), 'a', encoding='utf-8', newline='')
            wr = csv.writer(f)
            wr.writerow(
                [self.model_type, acc, SE, SP, PC, F1, JS, DC, self.lr, best_epoch, self.num_epochs, self.num_epochs_decay,
                 self.augmentation_prob])
            f.close()

if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--image_size', type=int, default=160)

    parser.add_argument('--image_channels', type=int, default=3)
    parser.add_argument('--output_channels', type=int, default=1)
    parser.add_argument('--num_epoch', type=int, default=300)
    parser.add_argument('--num_epoch_decay', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--augmentation_prob', type=float, default=0.4)

    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--val_step',type=int, default=2)

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_path', type=str, default='D:\Repositories/U-Net/models')
    parser.add_argument('--model_type',type=str, default='U_net')
    parser.add_argument('--train_path', type=str, default='D:\Repositories/U-Net/train/')
    parser.add_argument('--valid_path', type=str, default='D:\Repositories/U-Net/valid/')
    parser.add_argument('--test_path', type=str, default='D:\Repositories/U-Net/test/')
    parser.add_argument('--result_path', type=str, default='D:\Repositories/U-Net/results/')

    parser.add_argument('--cuda_idx', type=int, default=1)

    config = parser.parse_args()

    cudnn.benchmark = True   # 启动GPU

    root = 'D:\Repositories/U-Net/train/'
    print('\n root: ' + root)

    GT_root = root[:-1] + '_GT/'
    print('\n GT_root: ' + GT_root)

    img_path = list(map(lambda x: os.path.join(root, x), os.listdir(root)))

    one_image = img_path[0]

    file_name = one_image.split('/')[-1][:-len(".bmp")]
    GT_path = GT_root + file_name + '.bmp'
    print('file_name: ' + GT_path)

    # GT = Image.open('D:\Repositories/U-Net/train_GT/0.bmp')
    # plt.imshow(GT)
    # plt.axis('off')
    # plt.show()

    dataset = ImageFolder(root=root, image_size=160, mode='train', augmentation_prob=0.4)
    train_loader = data.DataLoader(dataset=dataset,
                                  batch_size=1,
                                  sampler=None,
                                  num_workers=8,
                                  )

    solver = Solver(config, train_loader)

    if config.mode == 'train':
        solver.train()


