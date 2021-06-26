#_*_ coding:utf-8_*_
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import warnings

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model.coordinate_descent import ConvergenceWarning
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.preprocessing import label_binarize
from sklearn import metrics

## 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False
# 拦截异常
warnings.filterwarnings(action = 'ignore', category=ConvergenceWarning)

## 数据加载
df = pd.read_csv('admissions.csv')
print(df.head())
print(df['admit'].value_counts())


# 3. 模型所需要的特征属性获取(构建特征属性矩阵X和目标属性Y)
X = df.iloc[:, 1:]
Y = df.iloc[:, 0]


# 4. 数据分割
X_train, X_test,Y_train,Y_test = train_test_split(X, Y, train_size=0.9,test_size=0.1, random_state=16)
print ("原始数据条数:%d；训练数据条数:%d；特征个数:%d；测试样本条数:%d" % (len(X), len(X_train), X_train.shape[1], X_test.shape[0]))


# 6. 算法/模型对象构建
models = [
    Pipeline([
            ('ss', StandardScaler()),
            ('poly', PolynomialFeatures()),
            ('LR', LogisticRegression(solver='liblinear'))
        ]),
    Pipeline([
            ('ss', StandardScaler()),
            ('KNN',KNeighborsClassifier())
        ])
]


# 7. 网格参数交叉验证
parameters1= {
    'poly__degree':[1,2,3,4,5],
    'LR__penalty':['l1','l2'],
    'LR__fit_intercept':[True,False],
    'LR__C':np.logspace(-2,2,20)
}
parameters2={
    'KNN__n_neighbors': [3, 5, 7],
    'KNN__weights': ['uniform', 'distance'],
    'KNN__algorithm': ['kd_tree', 'brute','ball_tree'],
    'KNN__leaf_size': [1, 2, 3]
}

algo1 = GridSearchCV(estimator=models[0], param_grid=parameters1, cv=5)
algo2 = GridSearchCV(estimator=models[1], param_grid=parameters2, cv=5)



# LR和KNN模型比较运行图表展示
titles = ['LR', 'KNN']
parameters=[parameters1,parameters2]
colors = ['go', 'bo']
plt.figure(figsize=(16,8), facecolor='w')
ln_x_test = range(len(X_test))

plt.plot(ln_x_test, Y_test, 'ro', markersize = 6, zorder=3, label=u'真实值')
for t in range(2):
    # 获取模型并设置参数
    # GridSearchCV: 进行交叉验证，选择出最优的参数值出来
    # 第一个输入参数：进行参数选择的模型，
    # param_grid： 用于进行模型选择的参数字段，要求是字典类型；cv: 进行几折交叉验证
    model = GridSearchCV(models[t], param_grid=parameters[t],cv=5, n_jobs=1)#五折交叉验证
    # 模型训练-网格搜索
    model.fit(X_train, Y_train)
    # 模型效果值获取（最优参数）
    print ("%s算法:最优参数:" % titles[t],model.best_params_)
    print ("%s算法:准确率=%.3f" % (titles[t], model.best_score_))
    # 模型预测
    y_predict = model.predict(X_test)
    # 画图
    plt.plot(ln_x_test, y_predict, colors[t],  markersize = 10+6*t, zorder=2-t, label=u'%s算法估计值,准确率=%.3f' % (titles[t],model.best_score_))
# 图形显示
plt.legend(loc = 'best')
plt.xlabel(u'数据编号', fontsize=18)
plt.ylabel(u'是否审批(0表示通过，1表示通过)', fontsize=18)
plt.title(u"留学申请预测",fontsize=20)
plt.show()







