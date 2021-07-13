import numpy as np
from icecream import ic
features_1 = [3247258,3245839,4384569,3454569,3458456]
features_2 = [0.3476,0.03427,0.34784593,0.00432785,0.02376]

# 归一化，等比例缩放,不改变分布
def normalize(x):
    x = np.array(x)
    return (x - min(x)) / (max(x) - min(x))

# 标准化，变为正态分布
def standarized(x):
    x = np.array(x)
    return (x-np.mean(x)) / (np.std(x))

# 哑编码
def one_hot(elements):
    es = list(set(elements))
    encoding = []
    for e in elements:
      zeros = [0]*len(es)
      zeros[es.index(e)] = 1
      encoding.append(zeros)
    return encoding

if __name__ == '__main__':
    res1 = normalize(features_1)
    res1 = normalize(features_2)
    ic(res1,np.mean(res1),np.std(res1))
    ic(res1,np.mean(res1),np.std(res1))
    print('========================')
    res1 = standarized(features_1)
    res2 = standarized(features_2)
    ic(res1,np.mean(res1),np.std(res1))
    ic(res1,np.mean(res1),np.std(res1))
    print('========================')
    ic(one_hot(['合肥','南京','上海','广州','合肥','北京','合肥','上海']))