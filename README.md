# HandWritingDigital_Feature
用vgg16、resnet18对mnist、svhn和usps提取特征

## Datasets

* [MNIST](http://yann.lecun.com/exdb/mnist/)
* [SVHN](http://ufldl.stanford.edu/housenumbers/)
* [USPS](https://www.kaggle.com/bistaumanga/usps-dataset)

## Components

* `mnist/`: mnist数据集
* `svhn/`: svhn数据集
* `usps.h5`: usps数据集



* `vgg_train_model.py`: mnist微调vgg16
* `resnet_train_model.py`: mnist微调resnet18
* `vgg.pkl`: vgg16模型参数
* `resnet.pkl`: resnet18模型参数



* `vgg_mnist.py`: vgg16提取mnist特征
* `vgg_svhn.py`: vgg16提取svhn特征
* `vgg_usps.py`: vgg16提取usps特征
* `vgg_mnist`: vgg16提取mnist的特征
* `vgg_svhn`: vgg16提取svhn的特征
* `vgg_usps`: vgg16提取usps的特征



* `resnet_mnist.py`: resnet18提取mnist特征
* `resnet_svhn.py`: resnet18提取svhn特征
* `resnet_usps.py`: resnet18提取usps特征
* `resnet_mnist`: resnet18提取mnist的特征
* `resnet_svhn`: resnet18提取svhn的特征
* `resnet_usps`: resnet18提取usps的特征

