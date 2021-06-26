# -- encoding:utf-8 --
"""
Create by ibf on 19/1/23
"""

import torch
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

if __name__ == '__main__':
    # 加载训练好的模型并且进行稍微的调整修改。
    vgg16 = models.vgg16(pretrained=True)
    # 修改所有的模型参数不允许更新
    for param in vgg16.parameters():
        param.requires_grad = False
    # 更改模型的后面一部分的代码
    for idx in range(len(vgg16.features) - 1, 9, -1):
        del vgg16.features[idx]
    # 因为手写数字图片是灰度图像，只有一个通道，所以第一层的卷积需要更改。
    # vgg16.features[0] = torch.nn.Conv2d(1, 64, 3, 1, 1)
    vgg16.classifier = torch.nn.Sequential(
        torch.nn.Linear(6272, 500),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.9),
        torch.nn.Linear(500, 10)
    )
    print(vgg16)

    # 1. 加载数据
    mnist_root_path = '../datas/mnist'
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # 转换为RGB图像（因为模型中要求输入的是3通道的）
        transforms.ToTensor(),  # 首先将图像Image对象转换为Tensor对象
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 做一个标准化转换（x-mean）/std

    ])
    # train_data = datasets.CIFAR10('../datas/cifar_10', transform=transform, download=True)
    train_data = datasets.MNIST(root=mnist_root_path,  # 给定本地磁盘路径
                                transform=transform,  # 给定对于特征属性X的转换操作
                                target_transform=None,  # 给定对于特征属性Y做什么转换操作
                                train=True, download=True)
    data_loader_train = DataLoader(dataset=train_data,
                                   batch_size=64, shuffle=True)

    model = vgg16.train()
    loss = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=1e-6)

    for epoch in range(10):
        running_loss = 0
        running_accuracy = 0

        for data in data_loader_train:
            # 获取当前批次数据
            x_train, y_train = data
            x_train, y_train = Variable(x_train), Variable(y_train)
            # 获取当前批次对应的预测值
            outputs = model(x_train)
            pred = torch.argmax(outputs, 1)

            # 将梯度值设置为0
            optimizer.zero_grad()

            # 定义损失函数，并且反向传播（没有更新）
            _loss = loss(outputs, y_train)
            _loss.backward()

            # 更新参数
            optimizer.step()

            # 记录一下日志信息
            running_loss += _loss.data.data
            running_accuracy += torch.sum(pred == y_train.data)
        print("Loss is:{:.4f}, Accuracy is:{:.4f}%".format(running_loss / len(train_data),
                                                           100 * running_accuracy / len(train_data)))
