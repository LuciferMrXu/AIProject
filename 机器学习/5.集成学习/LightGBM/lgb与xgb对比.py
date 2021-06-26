#_*_ coding:utf-8_*_
import lightgbm as lgb
import xgboost as xgb

# lgb分类的sklearn训练方式
model1=lgb.LGBMRegressor(
    boosting_type='gbdt',   # 选择基学习器
    num_leaves=16,       # 层数控制（最大叶子数量:2^n-1）
    learning_rate=0.75,  # 学习率（0-1）
    n_estimators=165,   # 基学习器的个数
    subsample_for_bin=1000  # 直方图索引抽样
)
model1.fit()
# lgb的原生数据读取和转化方式
lgb.Dataset()
# lgb的原生训练方式
lgb.train()



# xgb分类的sklearn训练方式
model2=xgb.XGBClassifier()
model2.fit()
# xgb的原生数据读取和转化方式
xgb.DMatrix()
# xgb的原生训练方式
xgb.train()

