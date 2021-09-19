from numpy.lib.shape_base import column_stack
import yfinance as yf
import pandas as pd
import numpy as np
from icecream import ic
import os
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
 # 通过选股脚本筛选出10支股票(PE<PE_mean,MC>MC_mean)
tickers = ['300747.XSHE','603893.XSHG','300782.XSHE','688981.XSHG','600118.XSHG',
            '002389.XSHE','603025.XSHG','300661.XSHE','603290.XSHG']
tickers = [x.replace('XSHG','SS').replace('XSHE','SZ') for x in tickers]
# ic(tickers)

# # 下载这些股票的真实数据
# df = pd.DataFrame()
# for ticker in tickers:
#   temp_df = yf.download(tickers = ticker,start='2020-01-01',end='2020-12-31')
#   temp_df['name'] = ticker
#   # 合并到一个df中
#   df = pd.concat([df,temp_df],axis=0)

# df.to_csv(os.path.join(BASE_DIR,'data.csv'))

df = pd.read_csv(os.path.join(BASE_DIR,'data.csv'),index_col='Date')

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
    # df['positions'] = np.where(df['diff']>df['dea'], 1, 0)

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

def portfolio_buy(port, purchase_day, stock_name, num):
    """
        port: 之前买过的股票
        purchase_day: 购买日期
        stock_name: 购买的股票
        num: 购买数量
    """
    # 如果不开市
    if purchase_day not in open_dates: 
        print('Market closed for today')        
        return port
    # 计算股票需要购买的金额
    stock_price = df[(df.name == stock_name) & (df.Date == purchase_day)].iloc[0]['Open']
    order_price = stock_price * num
    # 考虑portfolio 钱不够的问题
    if port['cash'] < order_price:
        # 没买成，原封不动
        return port
    
    # 购买成功，cash减少
    port['cash'] -=  order_price
    # 如果之前没有持有过这个股票
    if stock_name not in port.keys():
        port[stock_name] = num
        return port
    elif stock_name in port.keys():
        port[stock_name] += num
        return port
    else:
        print('Error')
        return port



def portfolio_sell(port, sell_day, stock_name, num):
    """
        port: 自己手上的股票情况
        sell_day: 卖出日期
        stock_name: 股票名称
        num: 卖出数量
    """
    # 如果不开市
    if sell_day not in open_dates: 
        print('Market closed for today')
        return port    
    
    # 计算卖出的股票金额
    stock_price = df[(df.name == stock_name) & (df.Date == sell_day)].iloc[0]['Close']
    order_price = stock_price * num    
    # 如果之前没有持有过这个股票
    if stock_name not in port.keys():
        # 没卖成，原封不动
        return port
    # 如果卖出的数量 > 手上持有的数量，没卖成
    if num > port[stock_name]:
        return port    
    # 卖成功了，减少股票数量，增加cash
    if stock_name in port.keys():
        port[stock_name] -= num
        port['cash'] += order_price
        return port

# 计算投资组合价值
def get_portfolio_value(port, evaluation_date):
    df = df_copy.copy()
    df = df.reset_index()
    df.rename(columns={'index':'Date'},inplace=True)
    if evaluation_date not in open_dates: 
        print('Market closed for today')    
        return 0
    # 总价值
    total_value = 0
    # 累加每支股票的value
    for stock in port.keys():
        if stock == 'cash':
            total_value += port['cash']
            continue
        elif stock.startswith('cash_'):
            continue
        # 找到evaluation_date时，该股票的price
        stock_price = df[(df.name == stock) & (df.Date == evaluation_date)].iloc[0]['Close']
        # 计算该股票的value
        position = stock_price * port[stock]
        total_value += position
    
    # 打印当前的portfolio
    print(port)
    return total_value


if __name__ == '__main__':
    # MACD简单有效，但是需要注意一个问题，就是：进入信号总是很晚，需要注意向下的均线
    # 可以采用12,26 也可以采用 10和21
    ma1 = 12
    ma2 = 26

    # 假设初始资金 10000
    portfolio = dict()
    portfolio['cash'] = 10000

    # 深拷贝
    df_copy = df.copy()
    ic(df_copy)


    for ticker in tickers:
        # 这个ticker初始化的cash
        portfolio['cash_'+ticker] = 10000/len(tickers)
        # 筛选这个ticker的数据
        data = df_copy[df_copy['name'] == ticker]
        # 计算position，signals，ma1，ma2
        df = signal_compute(data)
        # ic(df)
        df.index = pd.to_datetime(df.index)
        # 买入的时刻点
        signal_buy = df.loc[df['signals']==1].index
        signal_buy = pd.DataFrame(np.array(signal_buy),columns=['Date'])
        signal_buy['signal'] = 'buy'
        # 卖出的时刻点
        signal_sell = df.loc[df['signals']==-1].index
        signal_sell = pd.DataFrame(np.array(signal_sell),columns=['Date'])
        signal_sell['signal'] = 'sell'
        # 信号合并
        signal = pd.concat([signal_buy,signal_sell],axis=0)
        # 按照时间顺序排序
        signal.sort_values(by=['Date'],ascending=True,inplace=True)
        df = df.reset_index()
        df.rename(columns={'index':'Date'},inplace=True)
        # ic(ticker)
        # plot(df,ticker)
        
        # 执行交易策略
        for index,row in signal.iterrows():
            # 统计收盘价格
            temp_data = str(row['Date'])[0:10]
            # ic(row['Date'])
            stock_price = df[(df['name'] == ticker)&(df['Date'] == temp_data)].iloc[0]['Close']

            # 可以买卖的交易日
            open_dates = np.unique(df['Date'].apply(lambda x : str(x)[0:10]))
            if row['signal'] == 'buy':
                # 计算购买数量
                buy_num = int(portfolio['cash_'+ticker] / stock_price)
                # 对ticker购买buy_num股
                portfolio = portfolio_buy(portfolio,temp_data,ticker,buy_num)
                portfolio['cash_'+ticker] -= stock_price*buy_num
                print(f'{temp_data}买入{ticker}股票{buy_num}，价格{stock_price}')
            elif row['signal'] == 'sell':
                # 对ticker全部卖出
                portfolio['cash_'+ticker] += stock_price*portfolio[ticker]
                print(f'{temp_data}卖出{ticker}股票{portfolio[ticker]}，价格{stock_price}')
                portfolio = portfolio_sell(portfolio,temp_data,ticker,portfolio[ticker])

               
   
        # 计算最后一天的持有价值
        stock_price = df[(df['name'] == ticker)&(df['Date'] == '2020-12-30')].iloc[0]['Close']
        stock_value = portfolio[ticker] * stock_price
        print('股票价值{0} ==> {1}'.format(int(10000/len(tickers)),stock_value+portfolio['cash_'+ticker]))

    ic(get_portfolio_value(portfolio,'2020-12-30'))



