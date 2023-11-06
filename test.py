from torch import optim
import torch.nn.functional as F
from evaluation import *
from model import U_net
from torchvision import transforms as T
import argparse
import os
from torch.backends import cudnn
import random
from data_loader import get_loader
from solver import Solver
from PIL import Image

class Solver(object):
    def __init__(self,config, train_loader, valid_loader, test_loader):

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

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


    # def update_lr(self, g_lr, d_lr):
    #     for param_group in self.optimizer.param_groups:
    #         param_group['lr'] = lr


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


    def test(self):

        unet_path = os.path.join(self.model_path, 'U_net-%d-%.4f-%d-%.4f.pkl' % (
        self.num_epochs, self.lr, self.num_epochs_decay, self.augmentation_prob))

        # U-Net Train
        if os.path.isfile(unet_path):
            # Load the pretrained Encoder
            self.unet.load_state_dict(torch.load(unet_path))
            print('U_net is Successfully Loaded from %s' % (unet_path))

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
            for i, (images, GT) in enumerate(self.test_loader):
                images = images.to(self.device)
                SR = F.sigmoid(self.unet(images))
                SR = SR.squeeze(0)  # 去掉批次维度，使其变为(1, 496, 496)
                # 将SR的数据从Tensor类型转换为Pillow的Image对象
                SR_image = Image.fromarray((SR[0].cpu().detach().numpy() * 255).astype('uint8'))
                save_path = os.path.join(self.result_path, 'SR_image_%d.bmp' % (i))

                SR_image.save(save_path)


def main(config):
    cudnn.benchmark = True   # 启动GPU

#     创建目录
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)

    config.result_path = os.path.join(config.result_path, config.model_type)

    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)

#     定义参数
    lr = random.random()*0.0005+0.0000005   # 学习率
    epoch = 300

    augmentation_prob = random.random()*0.7
    decay_ratio = random.random()*0.8
    decay_epoch = int(epoch*decay_ratio)    # 通常用于控制学习率在训练深度学习模型时的调整策略。它表示在经过多少个训练周期（epochs）后，要减小学习率的值。

    print(config)

    train_loader = get_loader(image_path=config.train_path,
                              image_size=config.image_size,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              mode='train',
                              augmentation_prob=config.augmentation_prob)

    valid_loader = get_loader(image_path=config.valid_path,
                              image_size=config.image_size,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              mode='valid',
                              augmentation_prob=config.augmentation_prob)

    test_loader = get_loader(image_path=config.test_path,
                              image_size=config.image_size,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              mode='test',
                              augmentation_prob=config.augmentation_prob)

    solver = Solver(config, train_loader, valid_loader, test_loader)

    # Train and sample the images
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()



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

    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--model_path', type=str, default='D:\Repositories/U-Net/models')
    parser.add_argument('--model_type',type=str, default='U_net')
    parser.add_argument('--train_path', type=str, default='D:\Repositories/U-Net/train/')
    parser.add_argument('--valid_path', type=str, default='D:\Repositories/U-Net/valid/')
    parser.add_argument('--test_path', type=str, default='D:\Repositories/U-Net/test/')
    parser.add_argument('--result_path', type=str, default='D:\Repositories/U-Net/results/')

    parser.add_argument('--cuda_idx', type=int, default=1)

    config = parser.parse_args()
    main(config)