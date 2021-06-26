#_*_ coding:utf-8_*_
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm,datasets
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

# 导入鸢尾花数据集(只选2个维度)
iris=datasets.load_iris()
X=iris.data[:,:2]
y=iris.target
# print(x)
# print(y)


C=1.0
svc=svm.SVC(kernel='linear',C=C).fit(X,y)
rbf_svc=svm.SVC(kernel='rbf',gamma=0.7,C=C).fit(X,y)
poly_svc=svm.SVC(kernel='poly',gamma='scale',degree=3,C=C).fit(X,y)
lin_svc=svm.LinearSVC(C=C).fit(X,y)

h=0.02
x_min,x_max = X[:,0].min()-1 ,X[:,0].max()+1
y_min,y_max = X[:,1].min()-1 ,X[:,1].max()+1
xx,yy = np.meshgrid(np.arange(x_min,x_max,h),
                    np.arange(y_min,y_max,h)
                    )
titles=['SVM原生',
        '线性核函数',
        '高斯核函数',
        '多项式核函数']

for i,clf in enumerate((svc,lin_svc,rbf_svc,poly_svc)):
    plt.subplot(2,2,i+1)
    plt.subplots_adjust(wspace=0.4,hspace=0.4)

    Z=clf.predict(np.c_[xx.ravel(),yy.ravel()])
    Z=Z.reshape(xx.shape)

    plt.contourf(xx,yy,Z,cmap=plt.cm.coolwarm,alpha=0.8)
    plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.coolwarm)

    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(),xx.max())
    plt.ylim(yy.min(),yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])

plt.show()