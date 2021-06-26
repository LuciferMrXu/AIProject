import pandas as pd
from src.utl.FileLoader import *
import glob

def any1():
    data_path = "../data/"
    files_path = glob.glob(data_path+"*.csv")

    for file in files_path:
        print("文件的名字是：",file)
        load2DfHead(path = file)

#查看 历史购买表 和 新增购买表中的 用户id和商户id的重合情况
def any2():
    def series2set(s1,s2):
        return len(set(list(s1)) & set(list(s2)))
    his_path = "../data/historical_transactions.csv"
    new_path = "../data/new_merchant_transactions.csv"
    train_path = "../data/train.csv"
    test_path = "../data/test.csv"
    df1 = pd.read_csv(his_path)
    df2 = pd.read_csv(new_path)

    re_uid = series2set(df1["card_id"], df2["card_id"])
    re_mid = series2set(df1["merchant_id"], df2["merchant_id"])
    df1_uid = len(set(list(df1["card_id"])))
    df2_uid = len(set(list(df2["card_id"])))
    df1_mid = len(set(list(df1["merchant_id"])))
    df2_mid = len(set(list(df2["merchant_id"])))
    print("df1的用户总量为：%d,df2的用户总量为：%d" %(df1_uid,df2_uid))
    print("两者用户重合量：%d" %re_uid)
    print("df1的商户总量为：%d,df2的商户总量为：%d" %(df1_mid,df2_mid))
    print("两者商家重合量：%d" % re_mid)

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    print("train与df1的重合量：%d,与df2的重合量%d,总量为：%d" %(series2set(df_train['card_id'],df1["card_id"]),
                                          series2set(df_train['card_id'], df2["card_id"]),
                                                 len(set(list(df_train['card_id'])))
                                                 ))
    print("test与df1的重合量：%d,与df2的重合量%d,总量为：%d" %(series2set(df_test['card_id'],df1["card_id"]),
                                          series2set(df_test['card_id'], df2["card_id"]),
                                         len(set(list(df_test['card_id'])))
                                         ))

#为preprocession的str2num函数服务，手动生成编码表
def any3():
    his_path = "../data/historical_transactions.csv"
    df = pd.read_csv(his_path)
    for col in df.select_dtypes(include=["object"]).columns:
        print(df[col].value_counts())

if __name__=='__main__':
    any3()

