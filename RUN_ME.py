import argparse
import os
from torch.backends import cudnn
from data_loader import get_loader
from solver import Solver
import csv

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

        # 创建Train_log.csv和Validation_log.csv文件
        train_log_path = os.path.join(config.result_path, 'Train_log.csv')
        validation_log_path = os.path.join(config.result_path, 'Validation_log.csv')

        # 列名
        header = ["Epoch","Loss", "ACC", "SE", "SP", "PC", "F1", "JS", "DC"]

        # 写入列名到Train_log.csv
        with open(train_log_path, 'w', newline='') as train_log_file:
            train_log_writer = csv.writer(train_log_file)
            train_log_writer.writerow(header)

        # 写入列名到Validation_log.csv
        with open(validation_log_path, 'w', newline='') as validation_log_file:
            validation_log_writer = csv.writer(validation_log_file)
            validation_log_writer.writerow(header)

        # 读取Raw_log.csv
        raw_log_path = os.path.join(config.result_path, 'Raw_log.csv')

        with open(raw_log_path, 'r', newline='') as raw_log_file:
            raw_log_reader = csv.reader(raw_log_file)
            next(raw_log_reader)  # 跳过原始文件的标题行

            for i, row in enumerate(raw_log_reader, start=1):
                epoch, ACC, SE, SP, PC, F1, JS, DC = map(float, row)

                if i % 2 == 1:
                    # 奇数行，保存到Train_log.csv
                    with open(train_log_path, 'a', newline='') as train_log_file:
                        train_log_writer = csv.writer(train_log_file)
                        train_log_writer.writerow([epoch, ACC, SE, SP, PC, F1, JS, DC])
                else:
                    # 偶数行，保存到Validation_log.csv
                    with open(validation_log_path, 'a', newline='') as validation_log_file:
                        validation_log_writer = csv.writer(validation_log_file)
                        validation_log_writer.writerow([epoch, ACC, SE, SP, PC, F1, JS, DC])

    elif config.mode == 'test':
        solver.test(250)           # 修改epoch，以调用特定轮的模型参数

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_channels', type=int, default=3)
    parser.add_argument('--output_channels', type=int, default=1)
    parser.add_argument('--num_epoch', type=int, default=300)
    parser.add_argument('--num_epoch_decay', type=int, default=250)  # 通常用于控制学习率在训练深度学习模型时的调整策略。它表示在经过多少个训练周期（epochs）后，要减小学习率的值。
    parser.add_argument('--batch_size', type=int, default=1)        # !注意!如果使用data_loader.py的图像增强方法，则batch_size必须设置为1。
                                                                    # 因为该图像增强方法中存在random.randint()，它会使得各个batch_size的维度不同而报错。
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--augmentation_prob', type=float, default=0.4)

    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--val_step',type=int, default=2)

    parser.add_argument('--mode', type=str, default='test', help='train,test')
    parser.add_argument('--model_path', type=str, default='D:\Repositories/U-Net/models')
    parser.add_argument('--model_type',type=str, default='U_net')
    parser.add_argument('--train_path', type=str, default='D:\Repositories/U-Net/train/')
    parser.add_argument('--valid_path', type=str, default='D:\Repositories/U-Net/valid/')
    parser.add_argument('--test_path', type=str, default='D:\Repositories/U-Net/test/')
    parser.add_argument('--result_path', type=str, default='D:\Repositories/U-Net/results/')

    parser.add_argument('--cuda_idx', type=int, default=1)

    config = parser.parse_args()
    main(config)


