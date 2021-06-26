# -- encoding:utf-8 --
"""
Create by ibf on 2018/11/4
"""
from sklearn.externals import joblib


class Model_Loader(object):
    def __init__(self):
        # 1. 加载恢复模型
        self.scaler = joblib.load('./model/scaler.m')
        self.algo = joblib.load('./model/algo.m')

    def fetch_predict_value(self, x):
        return self.algo.predict(self.scaler.transform(x_test))


if __name__ == '__main__':
    # 1. 模型初始化
    model = Model_Loader()

    # 2. 使用恢复好的模型对需要预测的数据做一个预测即可
    x_test = [
        [4.216, 0.418],  # 18.4
        [5.360, 0.436],  # 23.0
        [3.666, 0.528]  # 15.8
    ]
    y_pred = model.fetch_predict_value(x_test)
    print("预测值为:{}".format(y_pred))
    x_test = [
        [4.448, 0.498],  # 19.6
        [3.270, 0.152]  # 13.8
    ]
    y_pred = model.fetch_predict_value(x_test)
    print("预测值为:{}".format(y_pred))
