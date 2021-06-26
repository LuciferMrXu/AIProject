# -*- coding=utf-8 -*-

#均线实例
from f_indictor import ma
from f_strategy import sp1
from CTPTrading import *
import time

#实例化
trading = CTPTrading('zn1806')#传入字符串格式的合约名称
#登录账号
trading.loginInitail()#
#订阅行情
trading.subcribeMd()#
#初始化参数
trading.initialList()
#加载交易策略
while(1):
    #返回1分钟数据数据, print trading.dfMinute trading.df(tick)
    trading.market3()
    print(trading.dfMinute)
    #这个位置可以加载合成任意周期分钟数据的函数
    pass
    #返回k-bar数据，print trading.Open, trading.High, trading.Low, trading.Close
    trading.market31()
    #加载策略
    if ((len(trading.df) > 2) and (len(trading.dfMinute) > 4)):
        #计算策略参数,样式如下
        Short = ma.maIndictor(trading.Close, 2, name='ma_5')
        Long = ma.maIndictor(trading.Close, 4, name = 'ma_10')
        #加载策略函数，样式如下
        pass
        Buy_, Sell_, Short_, Cover_ = sp1.sp_01(Short, Long)
        #开平仓信号判断，样式如下
        BuySignal = ((Buy_) and (trading.pos == 0))
        SellSignal = ((Sell_) and (trading.pos > 0 ))
        ShortSignal = ((Short_) and (trading.pos == 0))
        CoverSignal = ((Cover_) and (trading.pos < 0 ))
        print(BuySignal, SellSignal, ShortSignal, CoverSignal)
        print(Short.iloc[-2, 0])
        print(Long.iloc[-2, 0])
        print(trading.pos)#如果开仓信号没达到的时候，tading.pos一定为0
        #执行你的策略，开平仓接口样式如下：其中 200 为滑点，1为手数
        if BuySignal == True:
            trading.BuyOpen('zn1806', 200, 1)
            trading.BuyPosToday()
            trading.pos = trading.BuyPos
            continue
            pass#比如或可以加K线图绘制的代码
        if SellSignal == True:
            trading.SellClose('zn1806', 200, 1)
            trading.pos = 0
            continue
            pass
        if ShortSignal == True:
            trading.ShortOpen('zn1806', 200, 1)
            trading.ShortPosToday()
            trading.pos = trading.ShortPos*-1
            continue
            pass
        if CoverSignal == True:
            trading.CoverClose('zn1806', 200, 1)
            trading.pos = 0
            continue
            pass

    
    





