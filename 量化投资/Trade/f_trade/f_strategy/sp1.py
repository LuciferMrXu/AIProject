# -*- coding: utf-8 -*-

def sp_01(Short, Long, ):
    #加载你的策略
    BuyCondition = (Short.iloc[-2, 0]>Long.iloc[-2, 0])
    SellCondition = (Short.iloc[-2, 0]<Long.iloc[-2, 0])#可改为大于0或者等于1
    ShortCondition = (Short.iloc[-2, 0]<Long.iloc[-2, 0])
    CoverCondition = (Short.iloc[-2, 0]>Long.iloc[-2, 0])
    return BuyCondition, SellCondition, ShortCondition, CoverCondition
    