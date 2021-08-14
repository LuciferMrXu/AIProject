import pandas as pd
from sklearn.feature_extraction import DictVectorizer

train=pd.read_csv('train.csv',index_col=0)
test=pd.read_csv('test.csv',index_col=0)

#print(train['Attrition'].value_counts())
# 处理Attrition字段
train['Attrition']=train['Attrition'].map(lambda x:1 if x=='Yes' else 0)
from sklearn.preprocessing import LabelEncoder
# 查看数据是否有空值
#print(train.isna().sum())

# 去掉没用的列 员工号码，标准工时（=80）
train = train.drop(['EmployeeNumber', 'StandardHours'], axis=1)
test = test.drop(['EmployeeNumber', 'StandardHours'], axis=1)

# 对于分类特征进行特征值编码
attr=['Age','BusinessTravel','Department','Education','EducationField','Gender','JobRole','MaritalStatus','Over18','OverTime']
lbe_list=[]
for feature in attr:
    lbe=LabelEncoder()
    train[feature]=lbe.fit_transform(train[feature])
    test[feature]=lbe.transform(test[feature])
    lbe_list.append(lbe)
#print(train)
train.to_csv('train_label_encoder.csv')
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(train.drop('Attrition',axis=1), train['Attrition'], test_size=0.2, random_state=42)


model = LogisticRegression(max_iter=100, 
                           verbose=True, 
                           random_state=33,
                           tol=1e-4
                          )

model.fit(X_train, y_train)
predict = model.predict_proba(test)[:, 1]
test['Attrition']=predict

print(test['Attrition'])
test[['Attrition']].to_csv('submit_lr.csv')
print('submit_lr.csv saved')
# 转化为二分类输出
#test['Attrition']=test['Attrition'].map(lambda x:1 if x>=0.5 else 0)
#test[['Attrition']].to_csv('submit_lr.csv')
