# -*- coding: gbk -*-


# �������ݷ���
import pandas as pd
import numpy as np

# ���ڻ�ͼ
import matplotlib.pyplot as plt
import seaborn as sns


# ��ȡǰ��������
data_t = pd.read_csv('DATA/classifiy/titanic-data.csv')
print(data_t.head())
# ���ݼ���Ϣ���������ݼ���С������������
print(data_t.info())
print(data_t.columns.values)

# �ֶη���
def y(x):
    return data_t[x].unique()
print('='*20 + 'Survived�ֶ�����' + '='*20)
print(y('Survived'))
print('='*20 + 'Sex�ֶ�����' + '='*20)
print(y('Sex'))
print('='*20 + 'Pclass�ֶ�����' + '='*20)
print(y('Pclass'))
print('='*20 + 'Embarked�ֶ�����' + '='*20)
print(y('Embarked'))


# ��ʾ�ظ�����������
print(data_t.duplicated().value_counts())

# ��ʾ�п�ֵ����
print(data_t['Age'].isnull().value_counts())
print('-'*50)
print(data_t['Cabin'].isnull().value_counts())
print('-'*50)
print(data_t['Embarked'].isnull().value_counts())
print('-'*50)

# �����Է���
print(data_t.describe())

# �����ֵ
data_t['Age'] = data_t['Age'].fillna(data_t['Age'].mean()).astype(np.int64)
data_t['Embarked'] = data_t['Embarked'].fillna({"Embarked":"S"},inplace=True)
# ɾ���޹ص���
data_t = data_t.drop(['Ticket','Cabin'],axis='columns')
print(data_t.info())

#891�˵��У�����������δ���������Ƕ��٣�

total_survived = data_t['Survived'].sum()
total_no_survived = 891 - total_survived

plt.figure(figsize = (10,5)) # ��������
plt.subplot(121) # ��ӵ�һ����ͼ
sns.countplot(x='Survived',data=data_t)
plt.title('Survived count')

plt.subplot(122) # ��ӵڶ�����ͼ
plt.pie([total_survived,total_no_survived],labels=['Survived','No survived'],autopct='%1.0f%%')
plt.title('Survived rate')

plt.show()


# ��ͬ���������ֲ�
print(data_t.pivot_table(values='Name',index='Pclass',aggfunc='count'))
#print(data_t[['Pclass','Name']].groupby(['Pclass']).count()  )

plt.figure(figsize = (10,5)) # ��������
sns.countplot(x='Pclass',data=data_t)
plt.title('Person Count Across on Pclass')

plt.show()

#�ñ�ͼ

# plt.figure(figsize = (10,5)) # ��������
# plt.pie(data_t[['Pclass','Name']].groupby(['Pclass']).count(),labels=['1','2','3'],autopct='%1.0f%%')
# plt.axis("equal") #���Ʊ�׼��Բ��ͼ
#
# plt.show()

#��λ�������ʵĹ�ϵ


data_t.pivot_table(values='Survived',index='Pclass',aggfunc=np.mean)
plt.figure(figsize= (10 ,5))
sns.barplot(data=data_t,x="Pclass",y="Survived",ci=None) # ci��ʾ��������

plt.show()

## ��ͬ�Ա�������
data_t.pivot_table(values='Survived',index='Sex',aggfunc=np.mean)
plt.figure(figsize=(10,5))
sns.barplot(data=data_t,x='Sex',y='Survived',ci=None) 

plt.show()

#�ۺϿ����Ա�Sex������λ��Pclass���������ʹ�ϵ
data_t.pivot_table(values='Survived',index='Pclass',columns='Sex',aggfunc=np.mean)
plt.figure(figsize=(10,5))
sns.pointplot(data=data_t,x='Pclass',y='Survived',hue='Sex',ci=None)

plt.show()

#���䣨Age���������ʹ�ϵ
#
#data_t['Age'] =data_t.apply(lambda x:1  if  x['Age']==0  else x['Age'], axis=1)  # 

#for i  in  data_t['Age']:

#    print(i)
#bins=[0,20,40,60,80]
#data_t['AgeGroup'] = pd.cut(data_t['Age'],bins,include_lowest=True,labels=['baby', 'youth', 'adult', 'old']) 
#data_t['AgeGroup'] = pd.cut(data_t['Age'],5,labels=['baby', 'little','youth', 'adult', 'old']) # �����������ֵ����Ϊ��ȷ�
data_t['AgeGroup'] = pd.cut(data_t['Age'],5,labels=[1,2,3,4,5]) # �����������ֵ����Ϊ��ȷ�
#data_t['AgeGroup'] = pd.cut(data_t['Age'],5) # �����������ֵ����Ϊ��ȷ� �� ԭ�� �� ��� �� ����  ���� ��
#uu=0
#for i  in  data_t['AgeGroup']:
#    uu=uu+1
#    print(i,uu)

data_t.AgeGroup.value_counts(sort=False)

data_t.pivot_table(values='Survived',index='AgeGroup',aggfunc=np.mean)


plt.figure(figsize=(10,5))
sns.barplot(data=data_t,x='AgeGroup',y='Survived',ci=None)
plt.xticks(rotation=60) # ���ñ�ǩ�̶ȽǶ�

plt.show()


#�����ط���  ���䣨Age�����Ա�Sex���������ʹ�ϵ

data_t.pivot_table(values='Survived',index='AgeGroup',columns='Sex',aggfunc=np.mean)
print(data_t.pivot_table(values='Survived',index='AgeGroup',columns='Sex',aggfunc=np.mean))


plt.figure(figsize= (10 ,5))
sns.pointplot(data=data_t,x="AgeGroup",y="Survived",hue="Sex",ci=None,markers=["^", "o"], linestyles=["-", "--"])
# data_t.plot(kind='bar')
# plt.bar(left,height,width=width,color='lightskyblue',align='center',label='�Ͼ�')
plt.xticks(rotation=60)

plt.show()





