import argparse
import os
from torch.backends import cudnn
from data_loader import get_loader
from solver import Solver

def main(config):
    cudnn.benchmark = True   # 启动GPU

#     创建目录
    if not os.path.exists(config.model_path):     # 保存model参数的地址
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):    # 保存test结果图片的地址
        os.makedirs(config.result_path)

    config.result_path = os.path.join(config.result_path, config.model_type)

    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)

    print(config)

    train_loader = get_loader(image_path=config.train_path,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              mode='train',
                              augmentation_prob=config.augmentation_prob)

    valid_loader = get_loader(image_path=config.valid_path,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              mode='valid',
                              augmentation_prob=config.augmentation_prob)

    test_loader = get_loader(image_path=config.test_path,
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

    parser.add_argument('--image_channels', type=int, default=3)
    parser.add_argument('--output_channels', type=int, default=1)
    parser.add_argument('--num_epoch', type=int, default=300)
    parser.add_argument('--num_epoch_decay', type=int, default=250)  # 通常用于控制学习率在训练深度学习模型时的调整策略。它表示在经过多少个训练周期（epochs）后，要减小学习率的值。
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


