import os
from torch import optim
import torch.nn.functional as F
from evaluation import *
from model import U_net
from torchvision import transforms as T
from PIL import Image
import csv

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

    def reset_grad(self):
        """Zero the gradient buffers."""
        self.unet.zero_grad()

    def train(self, epoch = 0):
        """Train encoder, generator and discriminator."""

        # ====================================== Training ===========================================#
        # ===========================================================================================#

        unet_path = os.path.join(self.model_path, 'U_net-total_epoch_%d-lr_%.4f-num_epochs_decay_%d-augmentation_prob_%.4f' % (
                                        self.num_epochs, self.lr, self.num_epochs_decay, self.augmentation_prob))
        unet_path = os.path.join(unet_path + '-epoch_%dth.pkl' % (epoch))

        # U-Net Train
        if os.path.isfile(unet_path):                    # 如果已经存在该目录，则只会读取模型参数，不会进行训练。若要进行连续训练则要额外写代码
            # Load the pretrained Encoder
            self.unet.load_state_dict(torch.load(unet_path))
            print('U_net is Successfully Loaded from %s' % (unet_path))
        else:
            # Train for Encoder
            lr = self.lr

            # CSV文件地址
            csv_file = os.path.join(self.result_path, 'Raw_log.csv')
            with open(csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
            # 写入标题行
                writer.writerow(["Epoch", "ACC", "SE", "SP", "PC", "F1", "JS", "DC"])
                for epoch in range(self.num_epochs):

                    self.unet.train(True)
                    epoch_loss = 0

                    ACC = 0.  # Accuracy
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

                        # SR : Segmentation Result
                        SR = self.unet(images)
                        SR_probs = F.sigmoid(SR)
                        SR_flat = SR_probs.view(SR_probs.size(0), -1)

                        gray_transform = T.Grayscale(num_output_channels=1)
                        GT = gray_transform(GT)
                        width, height = SR.size(2), SR.size(3)
                        GT = torch.nn.functional.interpolate(GT, size=(width, height), mode='bilinear', align_corners=False)
                        GT_flat = GT.view(GT.size(0), -1)
                        loss = self.criterion(SR_flat, GT_flat)
                        epoch_loss += loss.item()

                        # Backprop + optimize
                        self.reset_grad()
                        loss.backward()
                        self.optimizer.step()

                        ACC += get_accuracy(SR, GT)
                        SE += get_sensitivity(SR, GT)
                        SP += get_specificity(SR, GT)
                        PC += get_precision(SR, GT)
                        F1 += get_F1(SR, GT)
                        JS += get_JS(SR, GT)
                        DC += get_DC(SR, GT)
                        length += images.size(0)

                    epoch_loss = epoch_loss / length
                    ACC = ACC / length
                    SE = SE / length
                    SP = SP / length
                    PC = PC / length
                    F1 = F1 / length
                    JS = JS / length
                    DC = DC / length

                    # 将指标数据写入 CSV 文件
                    data = [epoch, ACC, SE, SP, PC, F1, JS, DC]
                    writer.writerow(data)

                    # Print the log info
                    print(
                        'Epoch [%d/%d]\n [Training]\n Loss: %.4f, Average Accuracy: %.4f, Sensitivity: %.4f, Specificity: %.4f, '
                        'Precision: %.4f, F1: %.4f, Jaccard similarity: %.4f, Dice coefficient: %.4f' % (
                            epoch + 1, self.num_epochs,
                            epoch_loss,
                            ACC, SE, SP, PC, F1, JS, DC))

                    # Decay learning rate
                    if (epoch + 1) > (self.num_epochs - self.num_epochs_decay):
                        lr -= (self.lr / float(self.num_epochs_decay))
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = lr
                        print('Decay learning rate to lr: {}.'.format(lr))

                    if (epoch+1) % 10 == 0:
                        save_unet = self.unet.state_dict()
                        save_unet_path = os.path.join(self.model_path, 'U_net-total_epoch_%d-lr_%.4f-num_epochs_decay_%d-augmentation_prob_%.4f' % (
                                                     self.num_epochs, self.lr, self.num_epochs_decay, self.augmentation_prob))

                        save_unet_path = os.path.join(save_unet_path + '-epoch_%dth.pkl' % (i))
                        torch.save(save_unet, save_unet_path)

                    # ===================================== Validation ====================================#
                    self.unet.train(False)
                    self.unet.eval()


                    ACC = 0.  # Accuracy
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
                        SR = F.sigmoid(self.unet(images))

                        gray_transform = T.Grayscale(num_output_channels=1)
                        GT = gray_transform(GT)
                        width, height = SR.size(2), SR.size(3)
                        GT = torch.nn.functional.interpolate(GT, size=(width, height), mode='bilinear', align_corners=False)

                        ACC += get_accuracy(SR, GT)
                        SE += get_sensitivity(SR, GT)
                        SP += get_specificity(SR, GT)
                        PC += get_precision(SR, GT)
                        F1 += get_F1(SR, GT)
                        JS += get_JS(SR, GT)
                        DC += get_DC(SR, GT)

                        length += images.size(0)

                    ACC = ACC / length
                    SE = SE / length
                    SP = SP / length
                    PC = PC / length
                    F1 = F1 / length
                    JS = JS / length
                    DC = DC / length

                    # 将指标数据写入 CSV 文件
                    data = [epoch, ACC, SE, SP, PC, F1, JS, DC]
                    writer.writerow(data)

                    print('[Validation]\n Average Accuracy: %.4f, Sensitivity: %.4f, Specificity: %.4f, '
                          'Precision: %.4f, F1: %.4f, Jaccard similarity: %.4f, Dice coefficient: %.4f' % (
                    ACC, SE, SP, PC, F1, JS, DC))

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

                    # Save Best U-Net model                    在训练初期，这会导致保存大量的无用数据
                    # if unet_score > best_unet_score:
                    #     best_unet_score = unet_score
                    #     best_epoch = epoch
                    #     best_unet = self.unet.state_dict()
                    #     print('Best U_net model score : %.4f; Best epoch : %d' % (best_unet_score, best_epoch))
                    #
                    #     best_unet_path = os.path.join(self.model_path, 'U_net-total_epoch_%d-lr_%.4f-num_epochs_decay_%d-augmentation_prob_%.4f' % (
                    #                                  self.num_epochs, self.lr, self.num_epochs_decay, self.augmentation_prob))
                    #
                    #     best_unet_path = os.path.join(best_unet_path + '-Best_epoch_%d.pkl' % (best_epoch))
                    #     torch.save(best_unet, best_unet_path)

    def test(self, epoch):
        # ===================================== Test ====================================#
        unet_path = os.path.join(self.model_path, 'U_net-total_epoch_%d-lr_%.4f-num_epochs_decay_%d-augmentation_prob_%.4f' % (
                                        self.num_epochs, self.lr, self.num_epochs_decay, self.augmentation_prob))
        unet_path = os.path.join(unet_path + '-epoch_%dth.pkl' % (epoch))

        print(unet_path)

        # U-Net Train
        if os.path.isfile(unet_path):                    # 如果已经存在该目录，则只会读取模型参数，不会进行训练。若要进行连续训练则要额外写代码
            # Load the pretrained Encoder
            self.unet.load_state_dict(torch.load(unet_path))
            print('U_net is Successfully Loaded from %s' % (unet_path))

            self.build_model()
            self.unet.train(False)
            self.unet.eval()

            for i, (images, GT) in enumerate(self.test_loader):
                images = images.to(self.device)
                SR = F.sigmoid(self.unet(images))

                SR = SR.squeeze(0)  # 去掉批次维度，使其变为(1, 496, 496)
                # 将SR的数据从Tensor类型转换为Pillow的Image对象
                SR_image = Image.fromarray((SR[0].cpu().detach().numpy() * 255).astype('uint8'))
                save_path = os.path.join(self.result_path, 'SR_image_%d.bmp' % (i))

                SR_image.save(save_path)

        else:
            print('There is no U_net model')