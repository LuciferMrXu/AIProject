import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.preprocessing import PolynomialFeatures
outline_path = "../data/train_dataset.csv"
online_path = "../data/test_dataset.csv"


def load(path,sep):
    return pd.read_csv(path,sep=sep)

def clean(df,age_mid="",shop_mid=""):
    if not age_mid:
        age_mid = int(df['用户年龄'].median()) #36
    if not shop_mid:
        shop_mid = int(df['近三个月月均商场出现次数'].median()) #8
    df['用户年龄'][(df['用户年龄']>90) | (df['用户年龄']<10)] = age_mid
    df['近三个月月均商场出现次数'][df['近三个月月均商场出现次数'] > 30] = shop_mid
    return df,age_mid,shop_mid

def engineering(df):
    df['快递网购比'] = ratio_by0(df['当月物流快递类应用使用次数'],df['当月网购类应用使用次数'])
    df['月消费与历史消费比'] = ratio_by0(df['用户账单当月总费用（元）'],df['用户近6个月平均消费值（元）'])
    return df

def ratio_by0(s1,s2):
    ls = []
    for (a,b) in zip(s1,s2):
        if b == 0:
            ls.append(0.0)
        else:
            ls.append(a/b)
    return ls

def train(df):
    x = df.drop(['用户编码','信用分'],1)
    #poly = PolynomialFeatures(degree=2,interaction_only=True,include_bias=False)
    #x = poly.fit_transform(x)
    #尝试做01化
    y = df['信用分']
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=123)
    ds_train = lgb.Dataset(x_train, y_train)
    ds_test = lgb.Dataset(x_test, y_test, reference=ds_train)
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'l1',#mae
        'num_leaves': 32,
        #'max_depth': 6,
        #'min_data_in_leaf': 450,
        'learning_rate': 0.05,
        'feature_fraction': 0.7,#随机抽取特征
        'bagging_fraction': 0.95,#bagging抽取样本
        'bagging_freq': 5,#几次抽样放回
        'lambda_l1': 1,
        'lambda_l2': 1, # 越小l2正则程度越高
        #'min_gain_to_split': 0.2,
        #'is_unbalance': True
        }

    model = lgb.train(params, ds_train, num_boost_round=500, valid_sets=[ds_train, ds_test])
    return model

def predict(test_path,model,age_m,shop_m):
    df_test = load(test_path,",")
    uid = df_test['用户编码']
    df_test.drop(['用户编码'],1,inplace=True)
    df_test,_,_ = clean(df_test, age_m, shop_m)
    df_test = engineering(df_test)
    return model.predict(df_test), uid

def put_online(uid,score,output_path):
    ls = ["id,score"]
    with open(output_path,'w') as f:
        for (uids,scores) in zip(uid,score):
            ls.append(str(uids)+","+str(int(scores)))
        f.write("\n".join(ls))

def main():
    df = load(outline_path,sep=",")
    df,agem,shopm = clean(df)
    df = engineering(df)
    model = train(df)
    score,uid = predict(online_path,model,agem,shopm)
    put_online(uid, score, "../data/submit.csv")

if __name__ == "__main__":
    main()


