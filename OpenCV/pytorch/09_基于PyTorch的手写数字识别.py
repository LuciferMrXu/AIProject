# -- encoding:utf-8 --
"""
Create by ibf on 19/1/23
"""

import torch
from torch.autograd import Variable
import torch.nn.functional as F


class Model(torch.nn.Module):
    def __init__(self):
        # 调用父类的初始化相关方法
        super(Model, self).__init__()
        # 在这个位置进行网络的构建
        # 定义卷积（是否填充这个参数根据参数: padding来决定，padding等于0的话，就不填充）
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=20, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=20, out_channels=40, kernel_size=3, padding=1)
        # 定义全连接
        self.fc1 = torch.nn.Linear(in_features=7 * 7 * 40, out_features=120)
        self.fc2 = torch.nn.Linear(in_features=120, out_features=84)
        self.fc3 = torch.nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        """
        当我们构建好forward后，backward会自动构建。
        :param x:
        :return:
        """
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        # Reshape
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


if __name__ == '__main__':
    # 1. 构建网络对象
    net = Model()
    print(net)

    # 获取对象中的所有参数
    # print(list(net.parameters()))
    # print(list(net.named_parameters()))
    for name, param in net.named_parameters():
        print((name, param.size()))

    # 定义一个输入对象
    input_x = Variable(torch.randn([1, 1, 28, 28]))

    # 每次给定input_x的值，然后调用net模型即可
    out = net(input_x)
    print(out.size())
    print(out)
    print(torch.argmax(out, 1))
