import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.svm import OneClassSVM,SVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from ngboost import NGBClassifier
from ngboost.distns import k_categorical, Bernoulli
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler,Normalizer
import os
from icecream import ic

class StakingClassifier:
    def __init__(self,data):
        self.x = data.iloc[:,:-1]
        self.y = data.iloc[:,-1]

    def __call__(self):
        self.main()

    # 异常点检测
    def preprossion(self,x_train,x_test,y_train,y_test):
        one=OneClassSVM(kernel='rbf',gamma='scale',nu=0.1)
        one.fit(x_train)
        train_result=one.predict(x_train)
        test_result=one.predict(x_test)
        x_train = x_train[train_result==1]
        y_train = y_train[train_result==1]
        x_test = x_test[test_result == 1]
        y_test = y_test[test_result==1]
        return x_train,x_test,y_train,y_test

    # 划分数据集，并将划分好的数据用第一层单个模型拼接（纵向拼接）并输出预测值
    def stacking_step1(self,x_train,x_test,y_train,y_test,cv,model0):
        kf = KFold(n_splits=cv,shuffle=False)
        n = 0
        model_inner = []
        for train_index, test_index in kf.split(x_train):
            model = model0
            x_train_train, y_train_train = x_train[train_index],y_train[train_index]
            x_train_test,y_train_test = x_train[test_index],y_train[test_index]
            model.fit(x_train_train,y_train_train)
            model_inner.append(model)
            if n == 0:
                new_f_train = model.predict(x_train_test).reshape((-1,1))
                new_y_train = y_train_test.reshape((-1,1))
                new_f_test = model.predict(x_test).reshape((-1,1))
            else:
                new_f_train_tmp = model.predict(x_train_test).reshape((-1, 1))
                new_f_train = np.concatenate((new_f_train, new_f_train_tmp))
                new_y_train_tmp = y_train_test.reshape((-1,1))
                new_y_train = np.concatenate((new_y_train, new_y_train_tmp))
                new_f_test_tmp = model.predict(x_test).reshape((-1,1))
                new_f_test = new_f_test + new_f_test_tmp
            n += 1
        new_f_test = new_f_test / cv
        new_y_test = y_test
        return new_f_train, new_y_train, new_f_test, new_y_test, model_inner
    # 将第一层所有构建的模型集合都跑一遍（横向拼接），并输出的预测值
    def stacking_step2(self,x_train,x_test,y_train,y_test,cv,model_list):
        n = 0
        model_result = []
        for model in model_list:
            print("开始第%d个模型的训练"%(n+1))
            if n == 0:
                last_x_train, last_y_train, last_x_test, last_y_test,model_inner = self.stacking_step1(x_train,x_test,y_train,y_test,cv,model)
                model_result.append(model_inner)   # 接收第一步的所有模型
            else:
                new_f_train, new_y_train, new_f_test, new_y_test,model_inner = self.stacking_step1(x_train,x_test,y_train,y_test,cv,model)
                model_result.append(model_inner)
                last_x_train = np.concatenate((last_x_train,new_f_train),axis=1)
                last_x_test = np.concatenate((last_x_test,new_f_test),axis=1)
            n += 1
        return last_x_train, last_y_train, last_x_test, last_y_test, model_result
    # 第二层模型训练，传入第一层所有模型返回的预测值
    def stacking_step3(self,last_x_train, last_y_train, last_x_test, last_y_test, eval_result,last_model):
        # ic(last_x_train.shape)
        last_y_train=last_y_train.ravel()
        # ic(last_y_train.shape)
        last_model.fit(last_x_train,last_y_train)
        last_pred_train = last_model.predict(last_x_train)
        last_pred_test = last_model.predict(last_x_test)
        ic("train_result:",eval_result(last_y_train, last_pred_train))
        ic("test_result:",eval_result(last_y_test, last_pred_test))
        return last_model
    # 将一二层模型连接起来
    def stacking(self,x_train,x_test,y_train,y_test,cv,model_list,eval_result,last_model_l):
        last_model = last_model_l
        last_x_train, last_y_train, last_x_test, last_y_test, model_result = self.stacking_step2(x_train,x_test,y_train,y_test,cv,model_list)
        last_model = self.stacking_step3(last_x_train, last_y_train, last_x_test, last_y_test, eval_result, last_model)
        return model_result,last_model

    # 最终的stacking预测函数，对测试集做预测
    def predcit(self,x,cv,mode_list,last_model):
        i = 0
        for model_inner in mode_list:
            n = 0
            for model in model_inner:
                if n == 0:
                    new_x = model.predict(x.values).reshape((-1,1))
                else:
                    new_x = new_x + model.predict(x.values).reshape((-1,1))
                n += 1
            new_x = new_x / cv
            if i == 0:
                new_x_result = new_x
            else:
                new_x_result = np.concatenate((new_x_result,new_x),axis=1)
            i += 1
        y = last_model.predict(new_x_result)
        return y

    def main(self):
        y = LabelEncoder().fit_transform(self.y)
        # ic(y)
        x = StandardScaler().fit_transform(self.x)
        x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.75,test_size=0.25,random_state=16)
        x_train, x_test, y_train, y_test = self.preprossion(x_train,x_test,y_train,y_test)
        model_list = [
            XGBClassifier(max_depth=3,n_estimators=75,learning_rate=0.165,use_label_encoder=False,eval_metric=['logloss','auc','error']),
            LGBMClassifier(num_leaves=16,n_estimators=75,learning_rate=0.165),
            CatBoostClassifier(max_depth=5,n_estimators=75,learning_rate=0.165,verbose=False),
            NGBClassifier(Dist=k_categorical(2), verbose=False),
            RandomForestClassifier(max_depth=7,n_estimators=75),
            GradientBoostingClassifier(max_depth=5,n_estimators=75,learning_rate=0.165),
            AdaBoostClassifier(base_estimator=LogisticRegression(),learning_rate=0.165,n_estimators=75),
        ]
        last_model = SVC(kernel='rbf',gamma='auto',C=7)   # C越大越容易过拟合
        model_result, last_model_ed = self.stacking(x_train,x_test,y_train,y_test,5,model_list,accuracy_score,last_model)
        # online_df = pd.read_csv("../datas/zhengqi_test.txt",sep="\t")
        # online_pred = self.predcit(online_df,5,model_result,last_model_ed)
        # ic(online_pred)
        # with open('./zhengqi.txt','w') as f:
        #     online_list=[str(i) for i in online_pred]
        #     f.write('\n'.join(online_list))


if __name__=='__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(BASE_DIR,'voice.csv'))
    SC = StakingClassifier(data)
    SC()
