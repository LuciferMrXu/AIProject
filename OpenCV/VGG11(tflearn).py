# -*- coding: utf-8 -*-
from tflearn.datasets import oxflower17 
from tflearn.layers.core import dropout,input_data,fully_connected
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import tflearn
from sklearn.model_selection import train_test_split

X,Y=oxflower17.load_data(one_hot=True,resize_pics=(224,224))
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.75,test_size=0.25, random_state=16)

filter_size=[3,3]
kernel_size=[2,2]
stride=[2,2]


network=input_data(shape=[None,224,224,3])

network=conv_2d(network,nb_filter=64,filter_size=filter_size,activation='relu',name='block1_conv1')
network=local_response_normalization(network,name='block1_LRN')
network=max_pool_2d(network,kernel_size=kernel_size,strides=stride,name='block1_pool')


network=conv_2d(network,nb_filter=128,filter_size=filter_size,activation='relu',name='block2_conv1')
network=max_pool_2d(network,kernel_size=kernel_size,strides=stride,name='block2_pool')


network=conv_2d(network,nb_filter=256,filter_size=filter_size,activation='relu',name='block3_conv1')
network=conv_2d(network,nb_filter=256,filter_size=filter_size,activation='relu',name='block3_conv2')
network=max_pool_2d(network,kernel_size=kernel_size,strides=stride,name='block3_pool')


network=conv_2d(network,nb_filter=512,filter_size=filter_size,activation='relu',name='block4_conv1')
network=conv_2d(network,nb_filter=512,filter_size=filter_size,activation='relu',name='block4_conv2')
network=max_pool_2d(network,kernel_size=kernel_size,strides=stride,name='block4_pool')


network=conv_2d(network,nb_filter=512,filter_size=filter_size,activation='relu',name='block5_conv1')
network=conv_2d(network,nb_filter=512,filter_size=filter_size,activation='relu',name='block5_conv2')
network=max_pool_2d(network,kernel_size=kernel_size,strides=stride,name='block5_pool')


flatten_layer=tflearn.layers.core.flatten(network,name='flatten')


network=fully_connected(flatten_layer,4096,activation='relu')
network=dropout(network,0.5)
network=fully_connected(network,4096,activation='relu')
network=dropout(network,0.5)
network=fully_connected(network,1000,activation='relu')
network=dropout(network,0.5)

network=fully_connected(network,17,activation='softmax')

opt=tflearn.Momentum(learning_rate=0.1,lr_decay=0.1,decay_step=32000)
network=regression(network, optimizer=opt,loss='categorical_crossentropy')



model=tflearn.DNN(network,checkpoint_path='./model/model_vgg',tensorboard_dir='logs',max_checkpoints=3,tensorboard_verbose=3)
model.fit(x_train,y_train,n_epoch=50,validation_set=(x_test,y_test),shuffle=True,show_metric=True,batch_size=16,snapshot_step=200,snapshot_epoch=True,run_id='vgg')

 
