from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from icecream import ic
import numpy as np
from sklearn.metrics import r2_score
'''
    多个决策树模型组合成随机森林
'''

class RandomForest:
    def __init__(self,dataframe,y):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(dataframe, y, test_size=0.4)

    def decision_tree(self):
        regressor = DecisionTreeRegressor()

        regressor.fit(self.x_train, self.y_train)

        ic(regressor.score(self.x_train, self.y_train))
        ic(regressor.score(self.x_test, self.y_test))
        ic(r2_score(self.y_test, regressor.predict(self.x_test)))

    # 随机选择决策树训练的特征和训练数据个数
    def random_select(self,frame, y, drop=4):
        columns = np.random.choice(list(frame.columns), size=len(frame.columns) - drop)
        indices = np.random.choice(range(len(y)), size=len(y) - drop)

        return frame.iloc[indices][columns], y[indices]


    def random_forest(self):
        predicts = []

        for i in range(10):
            sample_x, sample_y = self.random_select(self.x_train, self.y_train)
            regressor = DecisionTreeRegressor()
            regressor.fit(sample_x, sample_y)
            train_score = regressor.score(sample_x, sample_y)
            test_score = regressor.score(self.x_test[sample_x.columns], self.y_test)
            ic(train_score)
            ic(test_score)
            # 把每一次的预测结果存下来
            predicts.append(regressor.predict(self.x_test[sample_x.columns]))
        # 将10个决策树的预测结果求平均值，即为随机森林的预测结果
        forest_predict = np.mean(predicts, axis=0)

        ic(r2_score(self.y_test, forest_predict))

if __name__ == '__main__':
    x = load_boston()['data']
    y = load_boston()['target']

    dataframe = pd.DataFrame(x, columns=load_boston()['feature_names'])

    rf = RandomForest(dataframe,y)
    rf.decision_tree()
    print('==========================================================')
    # 多个弱分类器组合成强分类器
    rf.random_forest()
