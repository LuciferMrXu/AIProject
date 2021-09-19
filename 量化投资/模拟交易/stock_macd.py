import yfinance as yf
from icecream import ic
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# tickers = ['AAPL', 'BABA', 'BIDU', 'GOOGL', 'FB', 'AMZN']
# data = yf.download(tickers = tickers, start = '2020-01-01', end = '2020-12-31')
# data = pd.DataFrame(data)
# ic(data)
# data.to_csv(os.path.join(BASE_DIR,'macd.csv'))

df = pd.read_csv(os.path.join(BASE_DIR,'macd.csv'))

# MACD简单有效，但是需要注意一个问题，就是：进入信号总是很晚，需要注意向下的均线
# 可以采用12,26 也可以采用 10和21
ma1 = 12
ma2 = 26

# 使用slicer进行切片，如果数据很大，回测曲线会比较乱，因为有太多的标记挤在一起
slicer = 0

# 计算df的 短期平均ma1, 长期平均ma2
def macd(df):
    # 计算MA1和MA2
    df['ma1']=df['Close'].rolling(window=ma1, min_periods=1).mean()   # 快线
    df['ma2']=df['Close'].rolling(window=ma2, min_periods=1).mean()   # 慢线
    # 短期均线 与 长期均线的DIFF
    df['diff']=df['ma1'] - df['ma2']
    # DEA = DIFF的平滑（9天的平均值）
    df['dea']=df['diff'].rolling(window=9, min_periods=1).mean()    # 两次平滑 ==> 过滤异常值效果好,去除噪音抖动
    return df


def signal_compute(df):
    """
        当短均线大于长均线时，我们看多并持有
        当短均线小于长均线时，我们清仓
        背后的逻辑是短均线有动量的影响（惯性）
        我们可以用 diff = 长均线-短均线
        diff有时正，有时负
        这就是为什么称为 Moving Average Convergence Divergence
    """
    # 计算短期平均ma1, 长期平均ma2, diff, dea
    df = macd(df)
    # 初始化positions均为0
    df['positions'] = 0

    # 当短均线 > 长均线， positions=1
    # df['ma1']快线特征，[mal:]12天后的数据
    df['positions'][ma1:] = np.where(df['ma1'][ma1:]>=df['ma2'][ma1:],1,0)
    #df['positions'] = np.where(df['diff']>df['dea'], 1, 0)

    # positions 表明了需要持有，计算前后两天的positions diff，代表交易信号 signals
    # signals=1 买入，signals=-1 卖出
    df['signals'] = df['positions'].diff()

    # 震荡diff = 两个移动平均之差
    df['diff'] = df['ma1']-df['ma2']
    df['macd'] = 2 * (df['diff'] - df['dea'])
    return df

# 绘制回测结果
def plot(df, ticker):    
    #the first plot is the actual close price with long/short positions
    # 绘制实际的股票收盘数据
    fig=plt.figure(figsize=(12, 6))
    ax=fig.add_subplot(111)    
    ax.plot(df.index, df['Close'], label=ticker)
    # 只显示时刻点，不显示折线图 => 设置 linewidth=0
    ax.plot(df.loc[df['signals']==1].index, df['Close'][df['signals']==1], label='Buy', linewidth=0, marker='^', c='g')
    ax.plot(df.loc[df['signals']==-1].index, df['Close'][df['signals']==-1], label='Sell', linewidth=0, marker='v', c='r')
    
    plt.legend(loc='best')
    plt.grid(True)
    plt.title('Positions')
    plt.show()
    
    # 显示diff, 即ma1-ma2
    fig=plt.figure(figsize=(12, 6))
    cx=fig.add_subplot(211)
    df['diff'].plot(kind='bar',color='r')
    #df['macd'].plot(kind='bar', color='r')
    
    plt.legend(loc='best')
    plt.grid(True)
    plt.xticks([]) # 不显示x轴刻度
    #plt.xlabel('')
    #plt.title('MACD Diff (ma1-ma2)')
    #plt.title('MACD 2*(diff-dea)')
    
    # 绘制ma1, ma2曲线
    bx=fig.add_subplot(212)
    bx.plot(df.index, df['ma1'], label='ma1')
    bx.plot(df.index, df['ma2'], label='ma2', linestyle=':')
   
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()







# 获取某一支股票的数据
# data = yf.download(tickers='AAPL', start='2020-01-01', end='2020-12-31')
# data.to_csv(os.path.join(BASE_DIR,'AAPL_macd.csv'))
# ic(data)

data = pd.read_csv(os.path.join(BASE_DIR,'AAPL_macd.csv'))



pd.set_option('max_rows', None)
# 计算ma1, ma2, positions, signals, diff指标
df = signal_compute(data)
df.index = pd.to_datetime(df.index)
ic(df)


print('买入信号:', df.loc[df['signals']==1].index)
print('买入信号时的收盘价', df['Close'][df['signals']==1])
print('卖出信号:', df.loc[df['signals']==-1].index)


plot(df, 'AAPL')
# 信号的分布
df['signals'].value_counts()
df['positions'].value_counts()

