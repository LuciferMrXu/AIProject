# -- encoding:utf-8 --
"""
Create by ibf on 19/1/23
"""

import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms


class AutoEncoder(torch.nn.Module):
    def __init__(self, with_cnn=False):
        # 调用父类的初始化相关方法
        super(AutoEncoder, self).__init__()

        self.with_cnn = with_cnn

        if self.with_cnn:
            # 定义编译器
            self.encoder = torch.nn.Sequential(
                torch.nn.Conv2d(1, 64, 3, 1, 1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(64, 128, 3, 1, 1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2)
            )
            # 定义解码器
            self.decoder = torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode='nearest'),
                torch.nn.Conv2d(128, 64, 3, 1, 1),
                torch.nn.ReLU(),
                torch.nn.Upsample(scale_factor=2, mode='nearest'),
                torch.nn.Conv2d(64, 1, 3, 1, 1)
            )
        else:
            # 定义编译器
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(28 * 28, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 32)
            )
            # 定义解码器
            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(32, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 28 * 28)
            )

    def forward(self, x):
        """
        当我们构建好forward后，backward会自动构建。
        :param x:
        :return:
        """
        # 格式转换一下
        if self.with_cnn:
            output = x.view(-1, 1, 28, 28)
        else:
            output = x.view(-1, 784)
        # 编码
        output = self.encoder(output)
        # 解码
        output = self.decoder(output)
        # 还原格式
        output = output.view(x.shape)
        return output


def train():
    # 1. 加载数据
    transform = transforms.Compose([
        transforms.ToTensor(),  # 首先将图像Image对象转换为Tensor对象
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 做一个标准化转换（x-mean）/std

    ])
    mnist_root_path = '../datas/mnist'
    train_data = datasets.MNIST(root=mnist_root_path,  # 给定本地磁盘路径
                                transform=transform,  # 给定对于特征属性X的转换操作
                                target_transform=None,  # 给定对于特征属性Y做什么转换操作
                                train=True, download=True)
    data_loader_train = DataLoader(dataset=train_data,
                                   batch_size=16, shuffle=True)

    # 1. 构建网络对象
    model = AutoEncoder(with_cnn=False)
    # 模型转换为训练阶段模型
    model = model.train()
    print(model)

    # 定义损失函数
    loss = torch.nn.MSELoss()
    optimizer = optim.Adam(params=model.parameters(), lr=1e-6)

    for epoch in range(10):
        running_loss = 0

        for data in data_loader_train:
            # 获取当前批次数据
            x_train, _ = data
            # 输入的是包含噪音数据的图像，输出希望是去除噪音数据后的图像。
            noisy_train = x_train + 0.5 * torch.randn(x_train.shape)
            x_train, noisy_train = Variable(x_train), Variable(noisy_train)
            # 获取当前批次对应的预测值(输入噪音数据，得到一个没有噪音的图像)
            outputs = model(noisy_train)

            # 将梯度值设置为0
            optimizer.zero_grad()

            # 定义损失函数，并且反向传播（没有更新）
            _loss = loss(outputs, x_train)
            _loss.backward()

            # 更新参数
            optimizer.step()

            # 记录一下日志信息
            running_loss += _loss.data.data
        print("Loss is:{:.4f}".format(running_loss / len(train_data)))

    # 做一个预测
    images, labels = next(iter(data_loader_train))
    print(labels)
    noisy_train = images + 0.5 * torch.randn(images.shape)
    pred_images = model(noisy_train).data
    # 将Tensor对象中的图像转换成网格图像的形式
    img = torchvision.utils.make_grid(pred_images)
    # 将img的Tensor对象转换为numpy; 并且将通道信息更改(c,h,w) -> (h,w,c)
    img = img.numpy().transpose(1, 2, 0)
    # 做一个反向的标准化
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    img = img * std + mean
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    train()
