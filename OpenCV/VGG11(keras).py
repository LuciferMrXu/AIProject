# -*- coding: utf-8 -*-
from tflearn.datasets import oxflower17
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import Dense,Activation,Conv2D,MaxPooling2D,Dropout,Flatten
from tensorflow.python.keras.models import Sequential,save_model,load_model
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from tensorflow.python.keras import optimizers

X,Y=oxflower17.load_data(one_hot=True,resize_pics=(224,224))

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.75,test_size=0.25, random_state=16)

learn_rate=0.01
batch_size=16
nb_epoch=10
img_rows,img_cols=224,224
pool_size=(2,2)        
kernel_size=(3,3)      
nb_classes=y_train.shape[1]


input_shape=(img_rows,img_cols,3)

model = Sequential()  
model.add(Conv2D(64,kernel_size,strides=(1,1),input_shape=input_shape,padding='same',kernel_initializer='uniform'))  
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size,strides=(2,2)))  

model.add(Conv2D(128,kernel_size,strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=pool_size,strides=(2,2)))  

model.add(Conv2D(256,kernel_size,strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(Conv2D(256,kernel_size,strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=pool_size,strides=(2,2)))

model.add(Conv2D(512,kernel_size,strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(Conv2D(512,kernel_size,strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=pool_size,strides=(2,2)))  

model.add(Conv2D(512,kernel_size,strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(Conv2D(512,kernel_size,strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=pool_size,strides=(2,2)))

model.add(Flatten())  
model.add(Dense(4096,activation='relu'))  
model.add(Dropout(rate=0.75))  
model.add(Dense(4096,activation='relu'))  
model.add(Dropout(rate=0.75))  
model.add(Dense(1000,activation='relu'))  
model.add(Dropout(rate=0.75)) 
model.add(Dense(nb_classes,activation='softmax'))

es=EarlyStopping(monitor='val_loss',patience=5)
reduce_lr=ReduceLROnPlateau(monitor='val_loss',patience=3,model='auto')
opt=optimizers.Adam(lr=learn_rate)
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

model.summary()

model.fit(x=x_train,y=y_train,batch_size=batch_size,epochs=nb_epoch,validation_data=(x_test,y_test))

save_model(model,'model/VGG11.model')

