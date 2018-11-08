# -*- coding: utf8 -*-
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
import torchvision.transforms as transforms
from torch.autograd import Variable

# 设置随机种子
torch.manual_seed(2)


EPOCH = 1 # 1个epoch就足够
BATCH_SIZE = 30 # batch大小
LR = 0.001 # 学习率
DOWNLOAD_MNIST = True  # 如果已经下载好了mnist数据就写上False


train_data = torchvision.datasets.MNIST(
    root='./mnist/',    # 保存或者提取位置
    train=True, # 训练集
    # 变换成vgg/resnet所需要的尺寸，再将像素值调整为[0,1]之间。
    transform=transforms.Compose([transforms.Resize(224),transforms.ToTensor(),transforms.Lambda(lambda x: x.repeat(3,1,1))]),
    download=DOWNLOAD_MNIST,
)
test_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=False, # 测试集
    transform=transforms.Compose([transforms.Resize(224),transforms.ToTensor(),transforms.Lambda(lambda x: x.repeat(3,1,1))]),
)

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)

# 读取模型
resnet = torchvision.models.resnet18(pretrained=True)
resnet.fc.out_features = 10

# 冻结最后一个残差块之前的网络层
for i,j in enumerate(resnet.parameters()):
    if i >= 45:
        break
    j.requires_grad = False

# 设置损失和优化器
resnet = resnet.cuda()
optimizer = torch.optim.Adam(resnet.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
        b_x = b_x.cuda()
        b_y = b_y.cuda()
        output = resnet(Variable(b_x))              # cnn output
        loss = loss_func(output, Variable(b_y))   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        if step % 50 == 0:
            all_accuracy = 0
            for t_step, (test_x, test_y) in enumerate(test_loader):
                test_x = test_x.cuda()
                test_y = test_y.cuda()
                test_output = resnet(Variable(test_x))
                pred_y = torch.max(test_output, 1)[1].data.cpu().squeeze().numpy()
                accuracy = float((pred_y == test_y.cpu().numpy()).astype(int).sum()) / float(test_y.size(0))
                all_accuracy += accuracy
            all_accuracy /= (t_step+1)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.2f' % all_accuracy)

# 保存模型参数
torch.save(resnet.state_dict(), 'resnet.pkl')
