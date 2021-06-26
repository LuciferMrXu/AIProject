#coding:utf-8
'''历史交易数据获取及投资参考数据'''
import tushare as ts
import matplotlib.pyplot as plt
import sys#调试  sys.exit('第X行')

# print ts.__version__

#获取股票历史基本信息
df = ts.get_k_data('399300', start= '2018-04-09', end = '2018-04-16', ktype = '15', autype = 'qfq')
# df = ts.get_hist_data('300085', ktype = '5')
#财报季抓取分红股
df = ts.profit_data(year = 2017, top= 100)
df.sort_values(by = 'divi', ascending = False, inplace = True)
#业绩预报选股
df = ts.forecast_data(2018, 2)
#限售解禁压力选股
df = ts.xsg_data()
df.sort_values(by = 'ratio', ascending = False, inplace = True)
#获得基金持股
df = ts.fund_holdings(2018, 1)
#获得新股数据
df = ts.new_stocks()

#融资融券信息查询
df = ts.sh_margins(start='2017-01-01', end='2018-04-16')
df = ts.sh_margins(start='2017-01-01', end='2018-04-16')
plt.plot(range(len(df)), df.loc[:,'rqyl'])
plt.show()
sys.exit(0)
#查询个股融资融券信息
df = ts.sh_margin_details(start='2017-01-01', end='2018-04-16', symbol='601989')
#查询深圳市场某日的融资融券信息
df = ts.sh_margin_details('2018-04-13')

print(df)
