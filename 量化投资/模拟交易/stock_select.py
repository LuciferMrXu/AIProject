"""
    使用JQData调取金融数据
"""
from datetime import datetime
import jqdatasdk as jq
import pandas as pd
from icecream import ic
jq.auth('18055121533', '1qa2ws3ed@XU')
# jq.auth('13282159964', '159964')



count=jq.get_query_count()
print("当日可调用数据总条数为：",count['total'])
print("当日剩余可调用条数为：",count['spare'])



# 获取平安银行股票数据，XSHE 深圳交易所 XSHG 上海交易所
test = jq.get_price("000001.XSHE", start_date="2021-01-01", end_date="2021-04-11")
ic(test)



# valuation是内置市值数据对象(这里表示查询valuation下所有的属性)
q = jq.query(jq.valuation).filter(jq.valuation.code=='000001.XSHE')
df = jq.get_fundamentals(q, '2021-01-01')
ic(df)


## 获取金融板块的股票
pd.set_option('max_rows', None)
# 汽车制造 C36，零售业 F52，其他金融业：J69, 货币金融服务：J66 
# 计算机、通信和其他电子设备制造业 C39
temp = jq.get_industries()
ic(temp)
# 筛选出和计算相关的产业
ic(temp[temp['name'].str.contains('计算机')])



"""
    智能投研
    1.筛选2020年市盈率小于行业平均水平
    2.市值大于行业平均水平的计算机行业股票
    3.然后观察2021年的市盈率表现
"""
#设置benchmark_date
benchmark_date = '2020-12-31'
# 设置股票范围，000018.XSHG 上证180金融股，XSHG 上海证券交易所
#pool = jq.get_index_stocks('000018.XSHG')

# 获取计算行业板块股票
"""
    pool1 = jq.get_industry_stocks('J69', date=benchmark_date)
    pool2 = jq.get_industry_stocks('J66', date=benchmark_date)
    # 得到所有金融业的股票
    pool = pool1 + pool2
"""
# 获取计算机行业板块股票
pool = jq.get_industry_stocks('C39', date=benchmark_date)
ic(pool)


# PE_ratio 15倍 
# 估值 = 每年利润 * 15倍
ic(len(pool))




# PE > 0,说明挣钱
q = jq.query(jq.valuation.code, jq.valuation.pe_ratio, jq.valuation.market_cap)\
    .filter(jq.valuation.pe_ratio > 0, jq.valuation.code.in_(pool))\
        .order_by(jq.valuation.pe_ratio.asc())
ic(q)

df = jq.get_fundamentals(q)
ic(df)
ic(df.shape)


"""
    查询股票代码 code, 市盈率pe_ratio, 市值market_cap
    1）市盈率>0
    2）在股票范围内（计算机行业）
    3）按照 PE从小到大排序
"""
q = jq.query(jq.valuation.code, jq.valuation.pe_ratio, jq.valuation.market_cap)\
    .filter(jq.valuation.pe_ratio > 0, jq.valuation.code.in_(pool))\
        .order_by(jq.valuation.pe_ratio.asc())

# 查询股票数据, 具体某一天date（之前最近的交易数据）
df = jq.get_fundamentals(q, date=benchmark_date)
# 计算平均PE，MC
pe_mean = float(df['pe_ratio'].mean())
mc_mean = float(df['market_cap'].mean())
print('满足条件的股票:{}'.format(len(df)))
print('平均PE:{} 平均MC:{}'.format(pe_mean,mc_mean))
ic(df)



"""
    筛选出来我们想要的股票
    1）小于PE平均，且PE>0（PE代表市场估值）
    2）大于MC平均（MCd代表市场规模）
    3）在股票范围（计算机行业）
    查询股票代码code, PE
    按照PE 从小到大排序
"""
q = jq.query(jq.valuation.code, jq.valuation.pe_ratio)\
    .filter(jq.valuation.pe_ratio<pe_mean, jq.valuation.market_cap>mc_mean,\
        jq.valuation.pe_ratio>0, jq.valuation.code.in_(pool))\
            .order_by(jq.valuation.pe_ratio.asc())
# 对于筛选出来的股票，查询benchmark_date的数据
df = jq.get_fundamentals(q, benchmark_date)
ic(df)

# 输出筛选出来的74支股票
print(df['code'].values)


# 从2021-1-1 到 2021-3-5中，这些股票的数据
start_date = datetime(2021, 1, 1)
end_date = datetime(2021, 3, 5)
# 获取指定的交易日范围
all_trade_days = jq.get_trade_days(start_date=start_date, end_date=end_date)
for i in all_trade_days:
    # 设置第i天的 PE数据
    df[i] = jq.get_fundamentals(q, i)['pe_ratio']
ic(df)

