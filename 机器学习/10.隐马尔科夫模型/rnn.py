#_*_ coding:utf-8_*_
import tensorflow as tf
import numpy as np
from data import batch_iter
from tensorflow.contrib import rnn
import pandas as pd


df=pd.r('./datas/SH600000.txt',sep='\t',encoding='gbk')

print(df)