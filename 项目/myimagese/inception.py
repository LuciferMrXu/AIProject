#神经网络抽取特征
import time,glob,re,sys,logging,os,tempfile
import numpy as np
import tensorflow as tf
from scipy import spatial

from tensorflow.python.platform import gfile
from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections
CONFIG_PATH = __file__.split('settings.py')[0]
INDEX_PATH = CONFIG_PATH+"index/"
DATA_PATH = CONFIG_PATH+"images/"

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='logs/worker.log',
                    filemode='a')

DIMENSIONS = 2048
PROJECTIONBITS = 16
ENGINE = Engine(DIMENSIONS, lshashes=[RandomBinaryProjections('rbp', PROJECTIONBITS,rand_seed=2611),
                                      RandomBinaryProjections('rbp', PROJECTIONBITS,rand_seed=261),
                                      RandomBinaryProjections('rbp', PROJECTIONBITS,rand_seed=26)])

#把加载网络
def load_network(png=False):
    with gfile.FastGFile('data/network.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        if png:
            png_data = tf.placeholder(tf.string, shape=[])
            decoded_png = tf.image.decode_png(png_data, channels=3)
            _ = tf.import_graph_def(graph_def, name='incept',input_map={'DecodeJpeg': decoded_png})
            return png_data
        else:
            _ = tf.import_graph_def(graph_def, name='incept')


#加载索引，供查询
def load_index():
    index,files,findex = [],{},0
    print ("Using index path : {}".format(INDEX_PATH+"*.npy"))
    for fname in glob.glob(INDEX_PATH+"*.npy"):
        print("fname:",fname)
        logging.info("Starting {}".format(fname))
        try:
            t = np.load(fname)
            if max(t.shape) > 0:
                index.append(t)
            else:
                raise ValueError
        except:
            logging.error("Could not load {}".format(fname))
            pass
        else:
            file=open(fname.replace(".feats_pool3.npy",".files"))
            for i,f in enumerate(file.readlines()):
                files[findex] = f.strip()
                ENGINE.store_vector(index[-1][i,:],"{}".format(findex))#索引的数据存到engin里面
                findex += 1
                print("加载索引")
            logging.info("Loaded {}".format(fname))
    index = np.concatenate(index)

    print("load的形状:",index.shape)
    return index,files

#n是找到最近的12张图片
#query_vector是查询图片的维度，是(1,2048)
#index是索引图片的维度(35000,2048)     files是文件名
def nearest(query_vector,index,files,n=12):
    temp = []
    dist = []
    logging.info("started query")
    for k in range(index.shape[0]):
        temp.append(index[k])
        if (k+1) % 50000 == 0:
            temp = np.transpose(np.dstack(temp)[0])
            dist.append(spatial.distance.cdist(query_vector,temp))
            temp = []
    if temp:
        temp = np.transpose(np.dstack(temp)[0])
        print("temp的形状",temp.shape)
        print("query的形状",query_vector.shape)
        dist.append(spatial.distance.cdist(query_vector,temp))
    dist = np.hstack(dist)

    ranked = np.squeeze(dist.argsort())
    print(ranked)
    logging.info("query finished")
    return [files[k] for i,k in enumerate(ranked[:n])]

#更快的找12个
def nearest_fast(query_vector,index,files,n=12):
    return [files[int(k)] for v,k,d in ENGINE.neighbours(query_vector)[:n]]


def get_batch(path,batch_size = 1000):
    """
    Args:
        path: directory containing images
    Returns: Generator with dictionary  containing image_file_nameh : image_data, each with size =  BUCKET_SIZE
    """
    path += "/train/7/*"
    image_data = {}
    logging.info("starting with path {}".format(path))

    for i,fname in enumerate(glob.glob(path)):
        try:
            image_data[fname] = gfile.FastGFile(fname, 'rb').read()
        except:
            logging.info("failed to load {}".format(fname))
            pass
        if (i+1) % batch_size == 0:
            logging.info("Loaded {}, with {} images".format(i,len(image_data)))
            yield image_data
            image_data = {}
    yield image_data

#对索引的数据进行存储
def store_index(features,files,count,index_dir):
    feat_fname = "{}/{}.feats_pool3.npy".format(index_dir,count)
    files_fname = "{}/{}.files".format(index_dir,count)
    logging.info("storing in {}".format(index_dir))
    with open(feat_fname,'wb') as feats:
        np.save(feats,np.array(features))
    with open(files_fname,'w') as filelist:
        filelist.write("\n".join(files))

#抽取特征
def extract_features(image_data,sess):
    #得到第三个pooling的特征，维度是2048维
    pool3 = sess.graph.get_tensor_by_name('incept/pool_3:0')
    features = []
    files = []
    print("imgetype",type(image_data))
    for fname,data in image_data.items():
        try:
            pool3_features = sess.run(pool3,{'incept/DecodeJpeg/contents:0': data})
            features.append(np.squeeze(pool3_features))
            files.append(fname)
        except:
            logging.error("error while processing fname {}".format(fname))
    return features,files

def load_test_index():
    index,files,findex = [],{},0
    input_path="C:/Users/11725/PycharmProjects/tensorflow/myimagese/test.pytest/index/"
    print ("Using index path : {}".format(input_path+"3.feats_pool3.npy"))
    fname=input_path+"3.feats_pool3.npy"
    logging.info("Starting {}".format(fname))
    try:
        t = np.load(fname)
        if max(t.shape) > 0:
            index.append(t)
        else:
            raise ValueError
    except:
        logging.error("Could not load {}".format(fname))
        pass
    else:
        file=open(fname.replace(".feats_pool3.npy",".files"))
        for i,f in enumerate(file.readlines()):
            files[findex] = f.strip()
            ENGINE.store_vector(index[-1][i,:],"{}".format(findex))#索引的数据存到engin里面
            findex += 1
        logging.info("Loaded {}".format(fname))
    index = np.concatenate(index)
    print("loadtest的形状:" , index.shape)
    return index

if __name__ == '__main__':
    index, files=load_index()
    test_index=load_test_index()
    file=nearest(test_index,index,files)
    #file =nearest_fast(test_index,index,files)
    filename = file[0]

    name = filename.split('\\')[1].split('_')
    wuti = name[0]
    if (wuti == '0'):
        print("这个可能是飞机")
    if (wuti == '1'):
        print("这个可能是汽车")
    if (wuti == '2'):
        print("这个可能是鸟")
    if (wuti == '3'):
        print("这个可能是猫")
    if (wuti == '4'):
        print("这个可能是鹿")
    if (wuti == '5'):
        print("这个可能是狗")
    if (wuti == '6'):
        print("这个可能是青蛙")
    print(file)
    #filename=file(0)
    #filename.




