#coding:utf-8
import numpy as np


class Nbayes(object):
    def __init__(self):
        self.vocabulary = []  # 词典
        self.idf = 0   # 词典的idf权重值
        self.td = 0   # 训练集的权值矩阵
        self.tdm = 0  # P(x|yi)
        self.Pcates = {}  # P(yi)是一个类别词典
        self.labels = []  # 对应每个文本分类，是一个外部导入的列表
        self.doclength = 0  # 训练集文本数
        self.vocablen = 0  # 词典词长
        self.testset = 0  # 测试集
        
     # 导入和训练数据集，生成算法必须的参数和·数据结构   
    def train_set(self, trainset, classVec):    
        self.cate_prob(classVec)    # 计算每个分类在数据集中的概率：P(yi)
        self.doclength = len(trainset)
        tempset = set()
        [ tempset.add(word) for doc in trainset for word in doc ]   # 生成词典
        self.vocabulary = list(tempset)
        self.vocablen = len(self.vocabulary)
        self.calc_wordfreg(trainset)   # 计算词频数据集
        self.build_tdm()  # 按分类累计向量空间的每维值：P(x|yi)
       
    # 计算在数据集中每个分类的概率：P(yi)   
    def cate_prob(self, classVec):   
        self.labels = classVec
        labeltemps = set(self.labels)  # 获取全部分类
        for labeltemp in labeltemps:
            # 统计列表中重复的分类：self.labels.count(labeltemp)
            self.Pcates[labeltemp] =float(self.labels.count(labeltemp))/float(len(self.labels))
            
    # 生成普通的词频向量
    def calc_wordfreq(self, trainset):
        self.idf = np.zeros([1,self.vocablen])  # 1*词典数
        self.tf = np.zeros([self.doclength, self.vocablen])  # 训练文件数*词典数
        for indx in range(self.doclength):   # 遍历所有样本
            for word in trainset[indx]:   # 遍历文本中的每个词
                self.tf[indx, self.vocabulary.index(word)] += 1  # 找到文本的词在词典中的位置 + 1
            for signleword in set(trainset[indx]):
                self.idf[0, self.vocabulary.index(signleword)] += 1 
                
    # 按分类累计向量空间的每维值：P(x|yi)   
    def build_tdm(self):
        self.tdm = np.zeros([len(self.Pcates), self.vocablen])   # 类别行*词典列
        sumlist = np.zeros([len(self.Pcates),1])  # 统计每个分类的总值
        for indx in range(self.doclength):
            self.tdm[self.labels[indx]] += self.tf[indx]  # 将同一类别的词向量空间值累加
            # 统计每个分类的总值（是一个标量）
            sumlist[self.labels[indx]] = np.sum(self.tdm[self.labels[indx]])
            self.tdm = self.tdm/sumlist  # 生成P(x|yi)
            
    # 将测试集映射到当前词典
    def map2vocab(self, testdata):
                
            
        
         
        
        
        
        