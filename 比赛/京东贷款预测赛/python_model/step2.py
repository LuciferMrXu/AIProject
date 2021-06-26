# -- encoding:utf-8 --
"""
Create by ibf on 2018/8/3
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. 加载数据
df = pd.read_csv('../data/data_full.csv')
print("总样本数目:{}".format(df.shape))
df.info()

# 2. 提取用于模型训练的数据（使用全部数据训练或者使用有值的部分数据训练）
train_df = df[df.loan_sum.notnull()]
# train_df = df
print("可用于训练的样本数目:{}".format(train_df.shape))
train_df.info()

# 3. 计算各个特征的缺失率
print("*" * 100)
loss_rate_df = train_df.select_dtypes(include=['float', 'int64']).describe().T. \
    assign(missing_pct=train_df.apply(lambda x: (len(x) - x.count()) / float(len(x))))['missing_pct']
print(loss_rate_df)
print("*" * 100)

# 4. 删除缺失率比较高的特征
train_df = train_df.drop('Velocity_days_FtoNowPerPlan_per_uid', axis=1)
# train_df.info()

# 5. 填充数据
train_df = train_df.fillna(value=0.0)
# train_df.info()

# 5. 数据划分
x = train_df.drop(['uid', 'loan_sum'], axis=1)
y = train_df['loan_sum']
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=28)
print("训练集样本数目:{}".format(x_train.shape))
print("测试集样本数目:{}".format(x_test.shape))

# 6. 模型训练
model = GradientBoostingRegressor(max_depth=6, alpha=0.7, n_estimators=100, learning_rate=0.05)
model.fit(x_train, y_train)

# 7. 查看模型效果
print("训练集RMSE:{}".format(np.sqrt(mean_squared_error(y_train, model.predict(x_train)))))
print("测试集RMSE:{}".format(np.sqrt(mean_squared_error(y_test, model.predict(x_test)))))
