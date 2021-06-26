from sklearn.preprocessing import Imputer,OneHotEncoder
from src.utl.FileLoader import load2DfHead
import pandas as pd
#字符串字段字典
strdic = {"authorized_flag":{"Y":1,"N":0},
          "category_1":{"Y":1,"N":0},
          "category_3":{"A":0,"B":1,"C":2}
              }
#时间字段
date = "purchase_date"

#处理缺失值
def clean(df, mode = 1):
    if mode == 1:
        df_result = df.dropna(how='any',axis=0)
        print("clean之前的样本数为：%d \n clean之后的样本数为：%d "
              %(df.shape[0],df_result.shape[0]))
        return df_result
    if mode == 2:
        im = Imputer(strategy="mean")
        df_trans = im.fit_transform(df)
        return im, df_trans
    if mode == 3:
        im = Imputer(strategy="most_frequent")
        df_trans = im.fit_transform(df)
        return im, df_trans

#处理购买数据表
#一、训练集上
#二、测试集上
#生成的编码表的样式： 列名1~属性1:0 属性2:1 属性3:2
#                     列名2~属性1:0 属性2:1 属性3:2
#用的时候生成一个字典 {列名1：{属性1:0，属性2:1，属性3:2},....}
#把字符串转换为数字

def str2num(df, encode_dict):
    for col in encode_dict.keys():
        df_tmp = df[col]
        encode_tmp = encode_dict[col]
        for fea in encode_tmp.keys():
            df_tmp[df_tmp == fea] = encode_tmp[fea]
        df[col] = df_tmp
    return df

#onehot处理
def pre_transaction(df, need_onehot, task="train", onehot_model=None):
    df_needonehot = df[need_onehot]
    df_other = df.drop(need_onehot, axis=1)
    if task == "train":
        onehot = OneHotEncoder()
        df_needonehot = onehot.fit_transform(df_needonehot)
    elif task == "test":
        onehot = onehot_model
        df_needonehot = onehot.transform(df_needonehot)
    df_result = pd.concat([df_needonehot,df_other],axis=1)
    return df_result

#聚合数据，提取关键信息
def groupby_uid(df):
    fea = df.groupby(["card_id","merchant_id"]).sum()
    print(fea)
    print(fea.index)

#时间函数
def time_func(df, df1):
    #df就是购买表
    #df1就是用户信息表
    df3 = pd.merge(left=df,right=df1,how="left",on="card_id")
    date1 = pd.to_datetime(df3['purchase_date'], format="%Y-%m-%d %H:%M:%S")
    date2 = pd.to_datetime(df3['first_active_month'], format="%Y-%m")
    date_tmp = date1 - date2
    #date_result是每次购买与首次注册的一个时间差
    date_result = date_tmp.apply(lambda x: x.days)
    id = df3['card_id']
    df4 = pd.DataFrame({"card_id":id,"date_between":date_result})
    客户第一次购买与注册的时间差 = df4.groupby(["card_id"]).min()
    客户每次购买与注册的平均时间差 = df4.groupby(["card_id"]).mean()
    return 客户每次购买与注册的平均时间差

def model_train(df, mdoel):
    return 0

#主函数
def main():
    df = pd.read_csv(his_path)
    df2 = pd.read_csv("")
    clean(df,1) #清洗
    time_func(df,df2) #时间字段单独处理
    str2num(df,encode_dict=strdic) #把字符串转化为数字
    pre_transaction(df,[]) #onehot处理
    groupby_uid(df) #聚合购买表
    #下面把几张票join起来，生成训练数据
    model_train(df,"xgboost") #训练模型



if __name__=='__main__':
    his_path = "../data/historical_transactions.csv"
    df = load2DfHead(his_path,1000)
    print(df.head())
    #groupby_uid(df)
    # time_func(df)

