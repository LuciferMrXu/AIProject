#_*_ coding:utf-8_*_
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost.sklearn import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression

outline=pd.read_csv("../../datas/zhengqi_train.txt",sep="\t")
online=pd.read_csv("../../datas/zhengqi_test.txt",sep="\t")

outline.drop(['target'],1,inplace=True)
outline['label']=0
online['label']=1
df=pd.concat([outline,online])
x=df.drop(['label'],1)
y=df['label']



lr=LogisticRegression(solver='liblinear')
lr.fit(x,y)
print(lr.predict(outline.drop(['label'],1)))
# 线下集属于线上集的预测概率
outproba=lr.predict_proba(outline.drop(['label'],1))[:,1]
print(sum(outproba>0.01))  # 通过调整概率值筛选数据


# 取出最像线上测试集的outline数据作为测试集
outline=pd.read_csv("../../datas/zhengqi_train.txt",sep="\t")
new_test=outline[outproba>0.01]
new_train=outline[outproba<=0.01]


x_train=new_train.drop(['target'],1)
y_train=new_train['target']
x_test=new_test.drop(['target'],1)
y_test=new_test['target']



models =  Pipeline([
            ('ss', StandardScaler()),
            # ('pca', PCA()),
            ('xgb', XGBRegressor(gpu_id=0,max_bin=16,tree_method='gpu_hist'))
])

params = {
    # 'pca__n_components':[3],
    'xgb__max_depth': [1],
    'xgb__learning_rate':[0.075],
    'xgb__reg_alpha':[1],
    'xgb__reg_lambda':[1],
    'xgb__n_estimators':[470]
}

algo = GridSearchCV(estimator=models, param_grid=params, cv=5)
algo.fit(x_train, y_train)
print("最优参数:{}".format(algo.best_params_))
y_pred_train=algo.predict(x_train)
y_pred_test = algo.predict(x_test)
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse=mean_squared_error(y_test,y_pred_test)
print('训练集上MSE为：%.3f\n测试集上MSE为：%.3f'%(train_mse,test_mse))


online_df = pd.read_csv("../../datas/zhengqi_test.txt",sep="\t")
online_y = algo.predict(online_df)
#这个写入会导致最后多一行的空行
with open("../../results/zhengqi_xgb.txt",'w') as f:
    for line in online_y:
        print(line)
        f.write(str(line)+"\n")