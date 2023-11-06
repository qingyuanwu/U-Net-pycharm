import os.path
from PIL import Image
import csv


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

# CSV文件地址
csv_file = os.path.join(result_path, 'Raw_log.csv')
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    # 写入标题行
    writer.writerow(["Epoch", "ACC", "SE", "SP", "PC", "F1", "JS", "DC"])
    data = [epoch, ACC, SE, SP, PC, F1, JS, DC]
    writer.writerow(data)
