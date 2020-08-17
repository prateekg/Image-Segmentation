from __future__  import print_function
import warnings
import torch
import cv2
import matplotlib.pyplot as plt
import os
import sys
import random

import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import models

#Seeding
seed = 2019
random.seed = seed
np.random.seed = seed


####################################### DataGenerator ##########################################
class DataGen(Dataset):
    def __init__(self, ids, path, batch_size=8, image_size=512):
        self.ids = ids
        self.path = path
        self.batch_size = batch_size
        self.image_size = image_size
        self.on_epoch_end()
    def __load__(self, id_name):
        image_path = os.path.join(self.path, id_name, "images", id_name) + ".png"
        mask_path = os.path.join(self.path, id_name, "masks/")
        all_masks = os.listdir(mask_path)
        ## Reading Image
        image = cv2.imread(image_path, 1)
        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = np.zeros((self.image_size, self.image_size, 1))
        ## Reading Masks
        for name in all_masks:
            _mask_path = mask_path + name
            _mask_image = cv2.imread(_mask_path, -1)
            _mask_image = cv2.resize(_mask_image, (self.image_size, self.image_size))  # 128x128
            _mask_image = np.expand_dims(_mask_image, axis=-1)
            mask = np.maximum(mask, _mask_image)
        ## Normalizaing
        image = image / 255.0
        mask = mask / 255.0
        return image, mask
    def __getitem__(self, index):
        if (index + 1) * self.batch_size > len(self.ids):
            self.batch_size = len(self.ids) - index * self.batch_size
        files_batch = self.ids[index * self.batch_size: (index + 1) * self.batch_size]
        image = []
        mask = []
        for id_name in files_batch:
            _img, _mask = self.__load__(id_name)
            image.append(_img)
            mask.append(_mask)
        image = np.array(image)
        mask = np.array(mask)
        return image, mask
    def on_epoch_end(self):
        pass
    def __len__(self):
        return int(np.ceil(len(self.ids) / float(self.batch_size)))


####################################### Model Helper ##########################################
def double_conv(in_channels, out_channels):
    conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )
    return conv

################################## When no padding is used. ###################################
# def crop_image(tensor, target_tensor):
#     target_size = target_tensor.size()[2]
#     tensor_size = tensor.size()[2]
#     delta = tensor_size-target_size
#     delta = delta//2
#     return tensor[:, :, delta:tensor_size-delta, delta:tensor_size-delta]

####################################### U-Net Model ###########################################
class UNet(nn.Module):
    def __init__(self, in_channel, out_class):
        super(UNet, self).__init__()
        #Unet Left Side
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv1 = double_conv(in_channel, 64)
        self.down_conv2 = double_conv(64, 128)
        self.down_conv3 = double_conv(128, 256)
        self.down_conv4 = double_conv(256, 512)
        self.down_conv5 = double_conv(512, 1024)
        self.up_trans1 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.up_conv1 = double_conv(1024, 512)
        self.up_trans2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.up_conv2 = double_conv(512, 256)
        self.up_trans3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.up_conv3 = double_conv(256, 128)
        self.up_trans4 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.up_conv4 = double_conv(128, 64)
        self.out = nn.Conv2d(64, out_class, 1)
    def forward(self, x):
        #bs, c, h, w
        #encoder
        c1 = self.down_conv1(x) #64*568*568
        #print(c1.size())
        x = self.max_pool(c1)
        c2 = self.down_conv2(x) #128*280*280
        #print(c2.size())
        x = self.max_pool(c2)
        c3 = self.down_conv3(x) #256*136*136
        #print(c3.size())
        x = self.max_pool(c3)
        c4 = self.down_conv4(x)  # 512*64*64
        #print(c4.size())
        x = self.max_pool(c4)
        c5 = self.down_conv5(x)  # 1024*28*28
        #decoder
        x = self.up_trans1(c5)
        #y = crop_image(c4, x)
        x = self.up_conv1(torch.cat([c4, x], 1))
        #print(x.size())
        x = self.up_trans2(x)
        #y = crop_image(c3, x)
        x = self.up_conv2(torch.cat([c3, x], 1))
        #print(x.size())
        x = self.up_trans3(x)
        #y = crop_image(c2, x)
        x = self.up_conv3(torch.cat([c2, x], 1))
        #print(x.size())
        x = self.up_trans4(x)
        #y = crop_image(c1, x)
        x = self.up_conv4(torch.cat([c1, x], 1))
        #print(x.size())
        x = self.out(x)
        return x


if __name__=="__main__":
    #Image Size -> 1x1x512x512
image_size = 512
train_path = "D:/Image Segmentation/data/data-science-bowl-2018/stage1_train/"
epochs = 5
batch_size = 8
train_ids = next(os.walk(train_path))[1]
val_data_size = 15
valid_ids = train_ids[:val_data_size]
train_ids = train_ids[val_data_size:]

train_gen = DataGen(train_ids, train_path, image_size=image_size, batch_size=batch_size)
valid_gen = DataGen(valid_ids, train_path, image_size=image_size, batch_size=batch_size)
train_steps = len(train_ids) // batch_size
valid_steps = len(valid_ids) // batch_size

in_channel = 3
out_class = 2
model = UNet(in_channel, out_class)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.99)

width_out = height_out = width_in = height_in = image_size
for epoch in epochs:
    total_loss = 0
    for i in range(train_steps):
        x, y = train_gen.__getitem__(i)
        outputs = model(torch.Tensor(x).permute(0, 3, 1, 2))
        outputs = outputs.permute(0, 2, 3, 1)
        m = outputs.shape[0]
        outputs = outputs.resize(m*width_out*height_out, 2)
        y = torch.Tensor(y).resize(m*width_out*height_out)
        loss = criterion(outputs, y.long())
        loss.backward()
        optimizer.step()
        print("Batch loss: %d", loss)
        total_loss += loss

    print("Average epoch Loss: %d", total_loss/train_steps)
