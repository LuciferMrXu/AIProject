# -- encoding:utf-8 --
"""
Create by ibf on 2018/10/21
"""

import re

from_email = '"pan" <pan@jdl.ac.cn>'
from_email = '"yan"<(8月27-28,上海)培训课程>'
# from_email = '=?GB2312?B?wfXOsNS0?= <qing@163.com>'
# from_email = 'wu@ccert.edu.cn'
it = re.findall(pattern=r"@([a-zA-Z0-9]+\.[a-zA-Z0-9\.]+)", string=from_email)
if len(it) > 0:
    result = it[0]
else:
    result = 'unknown'
print(result)

# 时间提取
date_str = "Mon 12 Sep 2005 11:16:56  +0800  X-Priority:  2  X-Mailer:"
date_str = "Sun 2 Oct 2005 04:18:16 -Mailer: OpenSmtp.net"
date_str = "Fri 2 Sep 2005 00:59:39 +0800 (CST)"
date_str = "Mon2Tue 27 Sep 2005 07:14:50 +0800"
it = re.findall(r"([A-Za-z]+\d?[A-Za-z]*) (\d{1,2}) ([A-Za-z]+) (\d+) (\d{2}):(\d{2}):(\d{2}).*", date_str)
print(it)

import jieba
jieba.add_word("北风网")
cut = jieba.cut("我来自北风网")
print(list(cut))

import pandas as pd

df = pd.DataFrame(data=[["我来自北风网"], ["我的名字叫小明"]], columns=['content'])
## 将文本类型全部转换为str类型，然后进行分词操作
df['content'] = df['content'].astype('str')
# jieba添加分词字典 jieba.load_userdict("userdict.txt")
df['jieba_cut_content'] = pd.Series(list(map(lambda st: "  ".join(jieba.cut(st)), df['content'])))
print(df.head(4))
