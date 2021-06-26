#Sequentical模型
# 实现一个lenet结构的网络
# 引入mnist数据
from tflearn.datasets import mnist
#from keras.datasets import mnist
# 引入网络结构的模块
from keras.layers import Dense,Activation,Conv2D,MaxPooling2D,Dropout,Flatten
from keras.models import Sequential,save_model,load_model
#from keras.utils import to_categorical
from keras.utils.multi_gpu_utils import multi_gpu_model
import keras



# 数据导入
X,Y,X_test,Y_test=mnist.load_data(one_hot=True)

# 定义超参数
batch_size=128
nb_classes=Y.shape[1]
nb_epoch=1

img_rows,img_cols=28,28
nb_step=2
pool_size=(2,2) # 池化的窗口大小
kernel_size=(5,5) # 卷积核的窗口大小

X=X.reshape([-1,img_rows,img_cols,1])
X_test=X_test.reshape([-1,img_rows,img_cols,1])
input_shape=(img_rows,img_cols,1)


# 将类向量转换为二进制矩阵
# Y=to_categorical(Y,nb_classes)
# Y_test=to_categorical(Y_test,nb_classes)

# 构建网络
model=Sequential()
# 卷积
model.add(Conv2D(64,kernel_size,padding='valid',input_shape=input_shape))
# 激励
model.add(Activation('relu'))

model.add(Conv2D(128,kernel_size))
model.add(Activation('relu'))
# 池化
model.add(MaxPooling2D(pool_size=pool_size))
# Dropout
model.add(Dropout(0.5))

# 全连接
# 数据拉伸称为2d
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))
# out输出
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# 损失函数和优化器的定义
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# 传入数据进行训练
model.fit(x=X,y=Y,batch_size=batch_size,epochs=nb_epoch,validation_data=(X_test,Y_test))
# model.fit_generator()
# 评估模型(损失和正确率)
score=model.evaluate(X_test,Y_test,verbose=0)
print("Test score：",score[0])
print("Test accuracy：",score[1])


# # 随机抽取数据做预测
# import numpy as np
# x=np.random.random((1,3))
# y=np.random.random((1,3,3))
# model.train_on_batch(x,y)
save_model(model,'cnn.model')
new_model=load_model('cnn.model')
# result=new_model.predict(x)

# # 模型的持久化和读取
# import tempfile
# # 大部分情况，使用tempfile模块
# # 把模型保存为.h5 文件，即HDFS 5 文件
# _,fname=tempfile.mkstemp('.h5')
# save_model(model,fname)
# new_model=load_model(fname)
new_result=new_model.predict(X_test[0])
