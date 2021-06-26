# -- encoding:utf-8 --
"""
Create by ibf on 2018/7/7
"""

import pandas as pd
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model.coordinate_descent import ConvergenceWarning
import xgboost as xgb
from sklearn.metrics import r2_score



warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
warnings.filterwarnings(action='ignore', category=RuntimeWarning)

# 1. 加载数据
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
file_path = '../datas/boston_housing.data'
df = pd.read_csv(filepath_or_buffer=file_path, header=None, sep='\\s+', names=names)

# 2. 获取特征矩阵X和目标属性Y
x = df[names[0:-1]]
y = df[names[-1]]

# 3. 数据的分割
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, test_size=0.1,random_state=28)
print("训练集大小:{}".format(x_train.shape))
print("测试集大小:{}".format(x_test.shape))

"""
开始使用XGBoost的相关API
"""
# 一、直接使用XGBoost的API
# a. 数据转换
dtrain = xgb.DMatrix(data=x_train, label=y_train)
dtest = xgb.DMatrix(data=x_test,label=y_test)

# b. 模型参数构建
params = {'max_depth': 10, 'eta': 1, 'objective': 'reg:linear','eval_metric':'rmse'}   # eta 是指每个模型的学习率
'''
    设置在GPU上运行
'''
params['gpu_id'] = 0
params['max_bin'] = 16
params['tree_method'] ='gpu_hist'



# c. 模型训练
model = xgb.train(params=params, dtrain=dtrain, num_boost_round=100,evals=[(dtrain, "train"),(dtest, "test")])
# d. 模型保存
model.save_model('xgb.model')
print(model)

# a. 加载模型产生预测值
model2 = xgb.Booster()
model2.load_model('xgb.model')
print(model2)
print("测试集R2：{}".format(r2_score(y_test, model2.predict(dtest))))
print("训练集R2：{}".format(r2_score(y_train, model2.predict(dtrain))))
