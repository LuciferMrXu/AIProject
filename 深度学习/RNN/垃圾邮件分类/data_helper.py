import re
import numpy as np
# 数据清洗，针对 \
def clean_str(string):
    string=re.sub(r"[A-Za-z9()!?\~]]"," ",string)# 对于单个数值，符号
    string=re.sub(r"\'s","\'s",string)# 把|'s 转换为's
    string=re.sub(r'\'ve','\'ve',string)
    string=re.sub(r'\'re','\'re',string)
    string=re.sub(r'\'t','\'t',string)
    string=re.sub(r'\'d','\'d',string)
    string=re.sub(r'\'ll','\'ll',string)
    string=re.sub(r',',' , ',string)
    string=re.sub(r'!',' ! ',string)
    string = re.sub(r'.', ' . ', string)
    return string.strip().lower()

# 读取数据，生成训练集
def load_data_and_label(positive_data_file,negative_data_file):
    # 读取数据
    positive=open(positive_data_file,'rb').read().decode('utf-8')
    negative=open(negative_data_file,'rb').read().decode('utf-8')

    # 生成样本，因为文本的倒数最后一行为空值，所以读取[0:-1]
    positive_examples=positive.split('\n')[:-1]
    negative_examples=negative.split('\n')[:-1]

    #清洗
    positive_examples=[clean_str(s).strip() for s in positive_examples]
    negative_examples=[clean_str(s).strip() for s in negative_examples]

    # 构建x和y
    x_text=positive_examples+negative_examples

    # y
    positive_label=[[0,1] for _ in range(len(positive_examples))]
    negative_label = [[1, 0] for _ in range(len(negative_examples))]
    y_label=np.concatenate([positive_label,negative_label],0)

    # 数据和标签合并到一起
    return [x_text,y_label]

# 第几步取batch_size行数据
# 直接返回每次迭代的数据（batch_size行） 下面选择这种形式
def batch_iter(data,batch_size,num_epochs,shuffle=True):
    data=np.array(data)
    data_size=len(data)

    # 一个epochs里面有多少个batch_size
    num_batches_epoch=(data_size-1)//batch_size+1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_index=np.random.permutation(np.arange(data_size))
            shuffle_data=data[shuffle_index]
        else:
            shuffle_data=data
        for batch_num in range(num_batches_epoch):
            start_index=batch_num*batch_size
            end_index=min(start_index+batch_size,data_size)
            yield shuffle_data[start_index:end_index]
