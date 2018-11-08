# -*- coding: utf8 -*-
import numpy as np
import torch
import torchvision
import torch.nn as nn
from scipy import io
import torch.utils.data as Data
import torchvision.transforms as transforms
from torch.autograd import Variable

torch.manual_seed(2)


BATCH_SIZE = 1

# 读取数据集
DOWNLOAD_MNIST = True  # 如果你已经下载好了mnist数据就写上False


train_data = torchvision.datasets.MNIST(
    root='./mnist/',    # 保存或者提取位置
    train=True,
    transform=transforms.Compose([transforms.Resize(224),transforms.ToTensor(),transforms.Lambda(lambda x: x.repeat(3,1,1))]),
    download=DOWNLOAD_MNIST,
)
test_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=False,
    transform=transforms.Compose([transforms.Resize(224),transforms.ToTensor(),transforms.Lambda(lambda x: x.repeat(3,1,1))]),
)

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

# 读取模型及参数
vgg = torchvision.models.vgg16(pretrained=True)
vgg.classifier[6].out_features = 10
vgg.load_state_dict(torch.load('vgg.pkl'))


# 剔除后面的全连接层，保留之前的卷积层用于提取特征
vgg = vgg.features
# vgg = nn.Sequential(*list(vgg.children())[:-1])
vgg = vgg.cuda()

# 训练集1提取
'''
train_features = torch.zeros((1, 512, 7, 7))
train_features = train_features.cuda()
train_labels = []
for i, (batch_data, batch_label) in enumerate(train_loader):
    if i >= 20000:
        break
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
io.savemat('mnist_train_features_1.mat', {'mnist_train_feature_1': train_feature})
io.savemat('mnist_train_labels_1.mat', {'mnist_train_label_1': train_label})
'''


# 训练集2提取
'''
train_features = torch.zeros((1, 512, 7, 7))
train_features = train_features.cuda()
train_labels = []
for i, (batch_data, batch_label) in enumerate(train_loader):
    if i < 20000:
        continue
    if i >= 40000:
        break
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
io.savemat('mnist_train_features_2.mat', {'mnist_train_feature_2': train_feature})
io.savemat('mnist_train_labels_2.mat', {'mnist_train_label_2': train_label})
'''


# 训练集3提取
'''
train_features = torch.zeros((1, 512, 7, 7))
train_features = train_features.cuda()
train_labels = []
for i, (batch_data, batch_label) in enumerate(train_loader):
    if i < 40000:
        continue
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
io.savemat('mnist_train_features_3.mat', {'mnist_train_feature_3': train_feature})
io.savemat('mnist_train_labels_3.mat', {'mnist_train_label_3': train_label})
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
io.savemat('mnist_test_features.mat', {'mnist_test_feature': test_feature})
io.savemat('mnist_test_labels.mat', {'mnist_test_label': test_label})
'''