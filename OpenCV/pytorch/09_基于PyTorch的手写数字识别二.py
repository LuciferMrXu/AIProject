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


class Model(torch.nn.Module):
    def __init__(self):
        # 调用父类的初始化相关方法
        super(Model, self).__init__()

        # 定义卷积层
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 20, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(20, 40, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )

        # 定义全连接层
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(7 * 7 * 40, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.85),
            torch.nn.Linear(1024, 10)
        )

    def forward(self, x):
        """
        当我们构建好forward后，backward会自动构建。
        :param x:
        :return:
        """
        # 卷积提取高阶特征
        x = self.conv(x)
        # 重置大小
        x = x.view(-1, 7 * 7 * 40)
        # 分类
        x = self.dense(x)
        return x


def train(mnist_root_path='../datas/mnist'):
    # 1. 加载数据
    transform = transforms.Compose([
        transforms.ToTensor(),  # 首先将图像Image对象转换为Tensor对象
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 做一个标准化转换（x-mean）/std

    ])
    train_data = datasets.MNIST(root=mnist_root_path,  # 给定本地磁盘路径
                                transform=transform,  # 给定对于特征属性X的转换操作
                                target_transform=None,  # 给定对于特征属性Y做什么转换操作
                                train=True, download=True)
    data_loader_train = DataLoader(dataset=train_data,
                                   batch_size=64, shuffle=True)

    # # 2. 查看一下
    # images, labels = next(iter(data_loader_train))
    # # 将Tensor对象中的图像转换成网格图像的形式
    # img = torchvision.utils.make_grid(images)
    # # 将img的Tensor对象转换为numpy; 并且将通道信息更改(c,h,w) -> (h,w,c)
    # img = img.numpy().transpose(1, 2, 0)
    # # 做一个反向的标准化
    # mean = [0.5, 0.5, 0.5]
    # std = [0.5, 0.5, 0.5]
    # img = img * std + mean
    # plt.imshow(img)
    # plt.show()

    # 1. 构建网络对象
    net = Model()
    # 模型转换为训练阶段模型
    net = net.train()
    print(net)

    # 获取对象中的所有参数
    # print(list(net.parameters()))
    # print(list(net.named_parameters()))
    # for name, param in net.named_parameters():
    #     print((name, param.size()))

    # # 定义一个输入对象
    # input_x = Variable(torch.randn([1, 1, 28, 28]))
    #
    # # 每次给定input_x的值，然后调用net模型即可
    # out = net(input_x)
    # print(out.size())
    # print(out)
    # print(torch.argmax(out, 1))

    # 定义损失函数
    loss = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=1e-6)

    for epoch in range(10):
        running_loss = 0
        running_accuracy = 0

        for data in data_loader_train:
            # 获取当前批次数据
            x_train, y_train = data
            x_train, y_train = Variable(x_train), Variable(y_train)
            # 获取当前批次对应的预测值
            outputs = net(x_train)
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

    # 模型持久化(要求输出的文件夹必须存在)
    # 方式一：将全部参数+网络对象持久化
    torch.save(net, './model/test09/01.pkl')
    # 方式二：仅仅保存参数，不保存网络对象
    torch.save(net.state_dict(), './model/test09/02.pkl')


def predict(mnist_root_path='../datas/mnist', type=1):
    # 1. 加载数据
    transform = transforms.Compose([
        transforms.ToTensor(),  # 首先将图像Image对象转换为Tensor对象
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 做一个标准化转换（x-mean）/std
    ])
    train_data = datasets.MNIST(root=mnist_root_path,  # 给定本地磁盘路径
                                transform=transform,  # 给定对于特征属性X的转换操作
                                target_transform=None,  # 给定对于特征属性Y做什么转换操作
                                train=True, download=True)
    data_loader_train = DataLoader(dataset=train_data,
                                   batch_size=4, shuffle=True)

    if type == 1:
        # 加载这个模型
        print("整个模型的加载....")
        model = torch.load('./model/test09/01.pkl')
    else:
        print("加载部分模型参数...<pytorch推荐>")
        model = Model()
        model.load_state_dict(torch.load('./model/test09/02.pkl'))
    # 模型转换为预测阶段模型(某些操作在预测过程中是不执行的)
    model = model.eval()

    # 做一个预测
    images, labels = next(iter(data_loader_train))
    pred = torch.argmax(model(images), 1)
    print("Real Label is:{}".format(labels))
    print("Predict Label is:{}".format(pred))
    # 将Tensor对象中的图像转换成网格图像的形式
    img = torchvision.utils.make_grid(images)
    # 将img的Tensor对象转换为numpy; 并且将通道信息更改(c,h,w) -> (h,w,c)
    img = img.numpy().transpose(1, 2, 0)
    # 做一个反向的标准化
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    img = img * std + mean
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    # train()
    predict(type=2)
