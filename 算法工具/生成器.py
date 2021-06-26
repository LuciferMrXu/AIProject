#_*_ coding:utf-8_*_
import numpy as np

x=[]
y=[]

for i in range(200):
    x.append(i)
    y.append(i)



def batch_iter(x, y, batch_size=10):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = np.array(x)[indices]
    y_shuffle = np.array(y)[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]

if __name__=='__main__':
    batch_train=batch_iter(x,y)
    for x_batch, y_batch in batch_train:
        print(x_batch,y_batch)