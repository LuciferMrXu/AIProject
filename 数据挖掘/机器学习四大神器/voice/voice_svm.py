import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC,LinearSVC
from sklearn.metrics import accuracy_score
# 数据加载
df = pd.read_csv('./数据挖掘/机器学习四大神器/voice/voice.csv')

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

# 选择模型训练
svc = SVC()
# svc = LinearSVC()
svc.fit(X_train,y_train)

# 预测
y_pred = svc.predict(X_test)

# 评估
acc = accuracy_score(y_test,y_pred)
print(acc)