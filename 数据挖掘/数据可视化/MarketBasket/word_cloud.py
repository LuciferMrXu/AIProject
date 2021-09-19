import pandas as pd
import numpy as np
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
from lxml import etree
from nltk.tokenize import word_tokenize
import jieba


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 去掉停用词
def remove_stop_words(f):
    stop_words = ['Movie']
    for stop_word in stop_words:
        f = f.replace(stop_word, '')
    return f

# 生成词云
def create_word_cloud(f):
    print('根据词频，开始生成词云!')
    f = remove_stop_words(f)
    cut_text = jieba.cut(f)
    # cut_text = word_tokenize(f)
    #print(cut_text)
    cut_text = " ".join(cut_text)
    wc = WordCloud(
      max_words=100,
      width=2000,
      height=1200,
      )
    wordcloud = wc.generate(cut_text)
    # 写词云图片
    wordcloud.to_file(os.path.join(BASE_DIR,"wordcloud.jpg"))
    # 显示词云文件
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()


if __name__ == '__main__':
  data = pd.read_csv(os.path.join(BASE_DIR,"Market_Basket_Optimisation.csv"),header=None)
  # 7501个样本，20个col
  print(data.shape)
  transactions=''
  # 行遍历
  for i in range(data.shape[0]):
      # 列遍历
      for j in range(data.shape[1]):
          if pd.isnull(data.values[i,j]) == False:
              transactions = transactions+' '+data.values[i,j]

  create_word_cloud(transactions)