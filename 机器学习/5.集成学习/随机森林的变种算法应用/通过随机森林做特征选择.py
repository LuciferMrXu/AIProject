#_*_ coding:utf-8_*_
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

'''
生成分类算法样本集，参数：
n_features :特征个数= n_informative（） + n_redundant + n_repeated
n_informative：多信息特征的个数
n_redundant：冗余信息，informative特征的随机线性组合
n_repeated ：重复信息，随机提取n_informative和n_redundant 特征
n_classes：分类类别
n_clusters_per_class ：某一个类别是由几个cluster构成的
weights:列表类型，权重比
class_sep:乘以超立方体大小的因子。 较大的值分散了簇/类，并使分类任务更容易。默认为1
'''
X,y=make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=3,
    n_redundant=0,
    n_repeated=0,
    n_classes=2,
    random_state=16,
    shuffle=False      # True是将原有序列打乱
)

# 构建250刻决策树的随机森林
forest=ExtraTreesClassifier(n_estimators=250,random_state=16)

forest.fit(X,y)

# 输出每个特征的重要性(根据每棵树的OOB计算袋外数据误差获得)
'''
选定一个feature M，在所有OOB样本的feature M上人为添加噪声，再测试模型在OOB上的判断精确率，
精确率相比没有噪声时下降了多少，就表示该特征有多重要。
假如一个feature对数据分类很重要，那么一旦这个特征的数据不再准确，对测试结果会造成较大的影响，
而那些不重要的feature，即使受到噪声干扰，对测试结果也没什么影响。
这就是 Variable importance 方法的朴素思想。
给特征X加噪声的方法:1.打乱顺序 2.在数据上加入白噪声
'''
importances=forest.feature_importances_
# print(importances)

# 求随机森林里所有决策树重要性的标准差
std=np.std(
    [tree.feature_importances_ for tree in forest.estimators_],
    axis=0
)
# 返回特征重要性从小到大的索引值
indices=np.argsort(importances)[::-1]
# print(indices)


# 按重要性从大到小排列特征值
print('Feature ranking:')
for f in range(X.shape[1]):
    print('%d.feature %d(%f)'%(f+1,indices[f],importances[indices[f]]))


# 画图
plt.figure()
plt.title('Feature importances')
plt.bar(
    range(X.shape[1]),
    importances[indices],
    color='r',
    yerr=std[indices],
    align='center'
)
plt.xticks(range(X.shape[1]),indices)
plt.xlim([-1,X.shape[1]])
plt.show()