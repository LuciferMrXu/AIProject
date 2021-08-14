import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score
import os
# 数据加载
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR,'voice.csv'))

print(df.head())
# 判断是否需要转换数据类型
print(df.info())

# 判断是否需要缺失值填充
print(df.isnull().sum())

# 判断样本大小
print(f'样本个数：{df.shape[0]}')
male = df[df['label']=='male'].shape[0]
print(f'男性个数：{male}')
female = df[df['label']=='female'].shape[0]
print(f'女性个数：{female}')


# 分离特征和target
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
print(y)


# 将y从str转为int
# y.apply(lambda x:1 if x=='male' else 0)

le = LabelEncoder()
y = le.fit_transform(y)
# male=>1 ,female=>0
print(y)


# 对x做归一化
scaler = StandardScaler()
X = scaler.fit_transform(X)
print(X)


# 切分训练集和测试集
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2021)

param = {'boosting_type':'gbdt',
                         'objective' : 'binary:logistic', #
                         'eval_metric' : 'auc',
                         'eta' : 0.01,
                         'max_depth' : 15,
                         'colsample_bytree':0.8,
                         'subsample': 0.9,
                         'subsample_freq': 8,
                         'alpha': 0.6,
                         'lambda': 0,
        }

train_data = xgb.DMatrix(X_train, label=y_train)
# valid_data = xgb.DMatrix(X_valid, label=y_valid)
test_data = xgb.DMatrix(X_test,label=y_test)

model = xgb.train(param, train_data, evals=[(train_data, 'train'),(test_data,'valid')], num_boost_round = 10000, early_stopping_rounds=200, verbose_eval=25)
predict = model.predict(test_data)
print(f'xgboost原始结果{predict}') # 概率值
predict = [1 if x>=0.5 else 0 for x in predict]
print(f'xgboost预测结果{predict}')
print(f'xgboost准确率{accuracy_score(y_test,predict)}')