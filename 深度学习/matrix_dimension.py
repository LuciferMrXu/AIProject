from torch import nn
import torch
import numpy as np
'''
    神经网络矩阵维度
'''

x = torch.from_numpy(np.random.random(size=(4, 100)))
print(x.shape)

model = nn.Sequential(
    nn.Linear(in_features=100, out_features=80).double(),
    nn.Sigmoid(),
    nn.Linear(in_features=80, out_features=70).double(),
    nn.Sigmoid(),
    nn.Linear(in_features=70, out_features=60).double(),
    nn.Sigmoid(),
    nn.Linear(in_features=60, out_features=50).double(),
    nn.Sigmoid(),
    nn.Linear(in_features=50, out_features=40).double(),
    nn.Sigmoid(),
    nn.Linear(in_features=40, out_features=30).double(),
    nn.Sigmoid(),
    nn.Linear(in_features=30, out_features=20).double(),
    nn.Sigmoid(),
    nn.Linear(in_features=20, out_features=10).double(),
    nn.Sigmoid(),
    nn.Linear(in_features=10, out_features=8).double(),
    nn.Sigmoid(),
    nn.Linear(in_features=8, out_features=4).double(),
    nn.Sigmoid(),
    nn.Linear(in_features=4, out_features=2).double(),
    nn.Softmax()
)
# 生成一个一行三列的随机数，取值范围(0-5)
print(torch.randint(5, (3, )))


ytrue = torch.randint(2, (4, ))
print(ytrue)

loss_fn = nn.CrossEntropyLoss()

print(model(x).shape)
print(ytrue.shape)
loss = loss_fn(model(x), ytrue)


loss.backward()

for p in model.parameters():
    print(p, p.grad)

