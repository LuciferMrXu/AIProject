import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from icecream import ic
import os

sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR,'all_stocks_5yr.csv'))
df.drop(['Unnamed: 0'], axis=1, inplace=True)
ic(df)
ic(df['name'].nunique())
ic(df.info())
ic(df.head())
ic(df.name.nunique())

open_dates = np.unique(df.Date)
ic(len(open_dates))
# 转行为pandas中的日期类型
df['Date'] = pd.to_datetime(df['Date'])
ic(df)


"""
    dates = df.date.apply(lambda x : pd.to_datetime(x, format='%Y-%m-%d', errors='ignore'))
    dates
    df = df.assign(day = dates) 
    df
"""
ic(df.dtypes)
ic(df.name.value_counts().head(5))
# 筛选苹果这支股票
sub_df = df[df.name == 'AAPL']
ic(sub_df.head())
sub_df.plot.line(x='Date', y='Open', figsize=(15,10))
plt.show()

# 计算投资组合价值
def get_portfolio_value(port, evaluation_date):
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
        # 找到evaluation_date时，该股票的price
        stock_price = df[(df.name == stock) & (df.Date == evaluation_date)].iloc[0]['Close']
        # 计算该股票的value
        position = stock_price * port[stock]
        total_value += position
    
    # 打印当前的portfolio
    print(port)
    return total_value


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



# 假设初始资金 10000
portfolio = dict()
portfolio['cash'] = 10000


ic(portfolio)

# 买入两支股票
portfolio_buy(portfolio, '2016-01-04', 'AAPL', 10)
portfolio_buy(portfolio, '2016-01-04', 'GOOG', 10)

get_portfolio_value(portfolio, '2016-01-04')
portfolio_sell(portfolio, '2017-02-01', 'AAPL', 20)
get_portfolio_value(portfolio, '2017-02-01')


# # 策略模拟
# 
# 假设我们模拟 2017-01-03 开始，初始资金 100000 元，大家各自选择购买策略, 比 2017年 最后一个交易日： 2017-12-29 的portfolio 价值。
# 
# ## 策略一： 2016-2017年，按照过去1年涨幅排序，直接全仓购买最好的
# 


# 筛选2016-2017年的股票数据
year_2016_df = df[
    (df.Date >= pd.to_datetime('2016-01-01', format='%Y-%m-%d', errors='ignore')) &
    (df.Date <= pd.to_datetime('2017-01-01', format='%Y-%m-%d', errors='ignore')) 
]
ic(year_2016_df)

best_stock = 'Not sure yet'
growth = -1

for stock in list(year_2016_df.name.unique()):
    # 计算每支股票的涨幅 stock_growth
    sub_df = year_2016_df[year_2016_df.name == stock]
    open_price = sub_df[sub_df.Date == min(sub_df.Date)].iloc[0]['Open']
    close_price = sub_df[sub_df.Date == max(sub_df.Date)].iloc[0]['Open']    
    stock_growth = round(100 * (close_price - open_price) / open_price, 3)
    print(stock, ':',  stock_growth)
    # 找到涨幅最高的股票
    if stock_growth > growth:
        best_stock = stock
        growth = stock_growth
# AMD 涨幅达到322.383%
print(best_stock, ':', growth)


# 假设初始资金 10000
portfolio = dict()
portfolio['cash'] = 10000


# 计算能买多少股
valid_num = int(portfolio['cash'] / df[(df.name == 'AMD') & (df.Date == '2017-01-03')].iloc[0]['Open'])
ic(valid_num)



# 在2017年初 重仓AMD
portfolio_buy(portfolio, '2017-01-03', 'AMD', valid_num)
# 计算2017年底 投资组合value
get_portfolio_value(portfolio, '2017-12-29')


# 统计2017年的交易日
trade_days_2017 = sorted(np.unique(df[
    (df.Date >= pd.to_datetime('2017-01-01', format='%Y-%m-%d', errors='ignore')) &
    (df.Date < pd.to_datetime('2018-01-01', format='%Y-%m-%d', errors='ignore'))     
]['Date'].apply(lambda x: str(x)[0:10])))

ic(trade_days_2017)


# 计算在2017年不同交易日的value
value = []
for trading_date in trade_days_2017:
    value.append(get_portfolio_value(portfolio, trading_date))

ic(value)


port_value_line = pd.DataFrame({'d': trade_days_2017, 'value':value})
# 在2017年 不同交易日的value变化
port_value_line.plot.line(x = 'd', y= 'value', figsize=(15,10))

plt.show()





