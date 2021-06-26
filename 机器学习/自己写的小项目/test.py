# -*- coding: gbk -*-


# 用于数据分析
import pandas as pd
import numpy as np

# 用于绘图
import matplotlib.pyplot as plt
import seaborn as sns


# 读取前五行数据
data_t = pd.read_csv('DATA/classifiy/titanic-data.csv')
print(data_t.head())
# 数据集信息，包含数据集大小，列名，类型
print(data_t.info())
print(data_t.columns.values)

# 字段分析
def y(x):
    return data_t[x].unique()
print('='*20 + 'Survived字段内容' + '='*20)
print(y('Survived'))
print('='*20 + 'Sex字段内容' + '='*20)
print(y('Sex'))
print('='*20 + 'Pclass字段内容' + '='*20)
print(y('Pclass'))
print('='*20 + 'Embarked字段内容' + '='*20)
print(y('Embarked'))


# 显示重复的数据数量
print(data_t.duplicated().value_counts())

# 显示有空值的列
print(data_t['Age'].isnull().value_counts())
print('-'*50)
print(data_t['Cabin'].isnull().value_counts())
print('-'*50)
print(data_t['Embarked'].isnull().value_counts())
print('-'*50)

# 描述性分析
print(data_t.describe())

# 处理空值
data_t['Age'] = data_t['Age'].fillna(data_t['Age'].mean()).astype(np.int64)
data_t['Embarked'] = data_t['Embarked'].fillna({"Embarked":"S"},inplace=True)
# 删除无关的列
data_t = data_t.drop(['Ticket','Cabin'],axis='columns')
print(data_t.info())

#891人当中，生还比率与未生还比率是多少？

total_survived = data_t['Survived'].sum()
total_no_survived = 891 - total_survived

plt.figure(figsize = (10,5)) # 创建画布
plt.subplot(121) # 添加第一个子图
sns.countplot(x='Survived',data=data_t)
plt.title('Survived count')

plt.subplot(122) # 添加第二个子图
plt.pie([total_survived,total_no_survived],labels=['Survived','No survived'],autopct='%1.0f%%')
plt.title('Survived rate')

plt.show()


# 不同船舱人数分布
print(data_t.pivot_table(values='Name',index='Pclass',aggfunc='count'))
#print(data_t[['Pclass','Name']].groupby(['Pclass']).count()  )

plt.figure(figsize = (10,5)) # 创建画布
sns.countplot(x='Pclass',data=data_t)
plt.title('Person Count Across on Pclass')

plt.show()

#用饼图

# plt.figure(figsize = (10,5)) # 创建画布
# plt.pie(data_t[['Pclass','Name']].groupby(['Pclass']).count(),labels=['1','2','3'],autopct='%1.0f%%')
# plt.axis("equal") #绘制标准的圆形图
#
# plt.show()

#舱位与生还率的关系


data_t.pivot_table(values='Survived',index='Pclass',aggfunc=np.mean)
plt.figure(figsize= (10 ,5))
sns.barplot(data=data_t,x="Pclass",y="Survived",ci=None) # ci表示置信区间

plt.show()

## 不同性别生还率
data_t.pivot_table(values='Survived',index='Sex',aggfunc=np.mean)
plt.figure(figsize=(10,5))
sns.barplot(data=data_t,x='Sex',y='Survived',ci=None) 

plt.show()

#综合考虑性别（Sex），舱位（Pclass）与生还率关系
data_t.pivot_table(values='Survived',index='Pclass',columns='Sex',aggfunc=np.mean)
plt.figure(figsize=(10,5))
sns.pointplot(data=data_t,x='Pclass',y='Survived',hue='Sex',ci=None)

plt.show()

#年龄（Age）与生还率关系
#
#data_t['Age'] =data_t.apply(lambda x:1  if  x['Age']==0  else x['Age'], axis=1)  # 

#for i  in  data_t['Age']:

#    print(i)
#bins=[0,20,40,60,80]
#data_t['AgeGroup'] = pd.cut(data_t['Age'],bins,include_lowest=True,labels=['baby', 'youth', 'adult', 'old']) 
#data_t['AgeGroup'] = pd.cut(data_t['Age'],5,labels=['baby', 'little','youth', 'adult', 'old']) # 将年龄的列数值划分为五等份
data_t['AgeGroup'] = pd.cut(data_t['Age'],5,labels=[1,2,3,4,5]) # 将年龄的列数值划分为五等份
#data_t['AgeGroup'] = pd.cut(data_t['Age'],5) # 将年龄的列数值划分为五等份 ， 原文 是 这个 ， 老是  报错 。
#uu=0
#for i  in  data_t['AgeGroup']:
#    uu=uu+1
#    print(i,uu)

data_t.AgeGroup.value_counts(sort=False)

data_t.pivot_table(values='Survived',index='AgeGroup',aggfunc=np.mean)


plt.figure(figsize=(10,5))
sns.barplot(data=data_t,x='AgeGroup',y='Survived',ci=None)
plt.xticks(rotation=60) # 设置标签刻度角度

plt.show()


#多因素分析  年龄（Age），性别（Sex）与生还率关系

data_t.pivot_table(values='Survived',index='AgeGroup',columns='Sex',aggfunc=np.mean)
print(data_t.pivot_table(values='Survived',index='AgeGroup',columns='Sex',aggfunc=np.mean))


plt.figure(figsize= (10 ,5))
sns.pointplot(data=data_t,x="AgeGroup",y="Survived",hue="Sex",ci=None,markers=["^", "o"], linestyles=["-", "--"])
# data_t.plot(kind='bar')
# plt.bar(left,height,width=width,color='lightskyblue',align='center',label='南京')
plt.xticks(rotation=60)

plt.show()





