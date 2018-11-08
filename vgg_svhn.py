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
DOWNLOAD_SVHN = True  # 如果你已经下载好了svhn数据就写上False


train_data = torchvision.datasets.SVHN(
    root='./svhn/',    # 保存或者提取位置
    split='train',
    transform=transforms.Compose([transforms.Resize(224),transforms.ToTensor()]),
    download=DOWNLOAD_SVHN,
)

test_data = torchvision.datasets.SVHN(
    root='./svhn/',
    split='test',
    transform=transforms.Compose([transforms.Resize(224),transforms.ToTensor()]),
)

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

# 读取模型及参数
vgg = torchvision.models.vgg16(pretrained=True)
vgg.classifier[6].out_features = 10
vgg.load_state_dict(torch.load('vgg.pkl'))


# 剔除后面的全连接层，保留之前的卷积层用于提取特征
vgg = nn.Sequential(*list(vgg.children())[:-1])
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
io.savemat('svhn_vgg_train_features_1.mat', {'svhn_train_feature': train_feature})
io.savemat('svhn_vgg_train_labels_1.mat', {'svhn_train_label': train_label})
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
io.savemat('svhn_vgg_train_features_2.mat', {'svhn_train_feature': train_feature})
io.savemat('svhn_vgg_train_labels_2.mat', {'svhn_train_label': train_label})
'''


# 训练集3提取
'''
train_features = torch.zeros((1, 512, 7, 7))
train_features = train_features.cuda()
train_labels = []
for i, (batch_data, batch_label) in enumerate(train_loader):
    if i < 40000:
        continue
    if i >= 60000:
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
io.savemat('svhn_vgg_train_features_3.mat', {'svhn_train_feature': train_feature})
io.savemat('svhn_vgg_train_labels_3.mat', {'svhn_train_label': train_label})
'''


# 训练集4提取
'''
train_features = torch.zeros((1, 512, 7, 7))
train_features = train_features.cuda()
train_labels = []
for i, (batch_data, batch_label) in enumerate(train_loader):
    if i < 60000:
        continue
    if i >= 80000:
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
io.savemat('svhn_vgg_train_features_4.mat', {'svhn_train_feature': train_feature})
io.savemat('svhn_vgg_train_labels_4.mat', {'svhn_train_label': train_label})
'''


# 测试集1提取
'''
test_features = torch.zeros((1, 512, 7, 7))
test_features = test_features.cuda()
test_labels = []
for i, (batch_data, batch_label) in enumerate(test_loader):
    if i >= 20000:
        break
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
io.savemat('svhn_vgg_test_features_1.mat', {'svhn_test_feature': test_feature})
io.savemat('svhn_vgg_test_labels_1.mat', {'svhn_test_label': test_label})
'''



# 测试集2提取
'''
test_features = torch.zeros((1, 512, 7, 7))
test_features = test_features.cuda()
test_labels = []
for i, (batch_data, batch_label) in enumerate(test_loader):
    if i < 20000:
        continue
    if i >= 40000:
        break
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
io.savemat('svhn_vgg_test_features_2.mat', {'svhn_test_feature': test_feature})
io.savemat('svhn_vgg_test_labels_2.mat', {'svhn_test_label': test_label})
'''
