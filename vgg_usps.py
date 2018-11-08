# -*- coding: utf8 -*-
import h5py
import numpy as np
import torch
import torchvision
import torch.nn as nn
from scipy import io
from PIL import Image
import torch.utils.data as Data
import torchvision.transforms as transforms
from torch.autograd import Variable

torch.manual_seed(2)


BATCH_SIZE = 1

# 读取USPS文件
with h5py.File('./usps.h5', 'r') as hf:
        train = hf.get('train')
        X_tr = train.get('data')[:]
        y_tr = train.get('target')[:]
        test = hf.get('test')
        X_te = test.get('data')[:]
        y_te = test.get('target')[:]

# 定义所需要的变换
transform=transforms.Compose([transforms.Resize(224),transforms.ToTensor(),transforms.Lambda(lambda x: x.repeat(3,1,1))])

# 将USPS定义成Dataset类
class USPS(Data.Dataset):
    def __init__(self, train=True, transforms=None):
        if train:
            self.imgs = X_tr
            self.labels = y_tr
        else:
            self.imgs = X_te
            self.labels = y_te
        self.imgs = self.imgs.reshape(self.imgs.shape[0], 16, 16)
        self.transforms = transforms
        
    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        if self.transforms:
            img = Image.fromarray(img)
            img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

train_data = USPS(train=True, transforms=transform)
test_data = USPS(train=False, transforms=transform)
train_loader = Data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = Data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# 读取模型及参数
vgg = torchvision.models.vgg16(pretrained=True)
vgg.classifier[6].out_features = 10
vgg.load_state_dict(torch.load('vgg.pkl'))

# 剔除后面的全连接层，保留之前的卷积层用于提取特征
vgg = nn.Sequential(*list(vgg.children())[:-1])
vgg = vgg.cuda()


# 训练集提取
'''
train_features = torch.zeros((1, 512, 7, 7))
train_features = train_features.cuda()
train_labels = []
for i, (batch_data, batch_label) in enumerate(train_loader):
    batch_data = Variable(batch_data.cuda())
    out = vgg(batch_data)
    out_np = out.data
    train_labels.append(int(batch_label[0]))
    train_features = torch.cat((train_features, out_np), 0)
    print(train_features.shape)
train_feature = train_features[1:,:]
train_feature = train_feature.cpu().numpy()
print(train_feature.shape)
train_label = np.matrix(train_labels)
io.savemat('usps_vgg_train_features.mat', {'usps_train_feature': train_feature})
io.savemat('usps_vgg_train_labels.mat', {'usps_train_label': train_label})
'''


# 测试集提取
'''
test_features = torch.zeros((1, 512, 7, 7))
test_features = test_features.cuda()
test_labels = []
for batch_data, batch_label in test_loader:
    batch_data = Variable(batch_data.cuda())
    out = vgg(batch_data)
    out_np = out.data
    test_labels.append(int(batch_label[0]))
    test_features = torch.cat((test_features, out_np), 0)
    print(test_features.shape)
test_feature = test_features[1:,:]
test_feature = test_feature.cpu().numpy()
print(test_feature.shape)
test_label = np.matrix(test_labels)
io.savemat('usps_vgg_test_features.mat', {'usps_test_feature': test_feature})
io.savemat('usps_vgg_test_labels.mat', {'usps_test_label': test_label})
'''