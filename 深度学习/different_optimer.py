import numpy as np
import torch
'''
    不同梯度下降法对比
'''
x = np.random.random(size=(100, 10))
# print(x)

linear = torch.nn.Linear(in_features=10, out_features=4)
sigmoid = torch.nn.Sigmoid()
linear2 = torch.nn.Linear(in_features=4, out_features=1)

model = torch.nn.Sequential(linear, sigmoid, linear2).double()

train_x = torch.from_numpy(x)

print(model(train_x).shape)

# uniform标准分布
yture = torch.from_numpy(np.random.uniform(0, 5, size=(100, 1)))


print(yture.shape)

loss_fn = torch.nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

for e in range(100):
    '''
        stochastic gradient descent(随机梯度下降，一次只训练一条数据，训练很快但是容易陷入局部最优解)
    '''
    for b in range(100 // 1): 
    # '''
    #     mini-batch gradient descent(小批量梯度下降，推荐)
    # '''    
    # for b in range(100 // 10):
    # '''
    #     batch gradient descent(批量梯度下降，把所有数据丢到内存中，太消耗资源，且训练很慢)
    # '''
    # for b in range(100 // 100):
        batch_index = np.random.choice(range(len(train_x)), size=20)

        yhat = model(train_x[batch_index])
        loss = loss_fn(yhat, yture[batch_index])
        loss.backward()
        print(loss)
        optimizer.step()
