# -- encoding:utf-8 --
"""
Create by ibf on 2018/10/20
"""

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

# 弱分类器的数目
n_estimator = 50
# 随机生成分类数据。
X, y = make_classification(n_samples=80000)
# 切分为测试集和训练集，比例0.5
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
# 将训练集切分为两部分，一部分用于训练RF模型，另一部分输入到训练好的RF模型生成RF特征，然后作为LR的输入特征。这样分成两部分是为了防止过拟合。
x_train, x_train_lr, y_train, y_train_lr = train_test_split(x_train, y_train, test_size=0.5)

# 1. 构建随机森林来训练模型
rf = RandomForestClassifier(n_estimators=n_estimator, max_depth=5)
rf.fit(x_train, y_train)

# 2. 获取训练LR模型的数据在RF上是落在哪些叶子节点的，以叶子节点的索引作为新的特征属性的值
x_train_lr1 = rf.apply(x_train_lr)

# 3. 对叶子的编号做一个哑编码操作
one_hot = OneHotEncoder()
one_hot.fit(rf.apply(x_train))
x_train_lr2 = one_hot.transform(x_train_lr1)

# 4. 做LR的训练
print("用于LR训练的数据特征:{}".format(x_train_lr2.shape))
lr = LogisticRegression()
lr.fit(x_train_lr2, y_train_lr)

# 5. 预测
y_pred = lr.predict(one_hot.transform(rf.apply(x_test)))
fpr, tpr, _ = roc_curve(y_test, y_pred)
print("AUC:{}".format(auc(fpr, tpr)))
