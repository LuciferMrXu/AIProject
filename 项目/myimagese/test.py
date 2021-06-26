import os,sys,logging,time,shutil
import inception
import glob
from tensorflow.python.platform import gfile

CONFIG_PATH = __file__.split('settings.py')[0]

def test():
    inception.load_network()
    count = 0
    start = time.time()
    test_images =  os.path.join(CONFIG_PATH+'test/images/*')
    print('test_images:',test_images)
    # for i, fname in enumerate(glob.glob(test_images)):
    #     image_data[fname] = gfile.FastGFile(fname, 'rb').read()
    #     print(fname)
    #
    #     image_data[fname] = gfile.FastGFile(fname, 'rb').read()
    #
    #     if (i+1) % 2 == 0:
    #         logging.info("Loaded {}, with {} images".format(i,len(image_data)))
    #         yield image_data
    #         image_data = {}
    # print(image_data)

    test_index = os.path.join(CONFIG_PATH+'test/index/')
    try:
        shutil.rmtree(test_index)
    except:
        pass
    print(test_index)
    os.mkdir(test_index)
    with inception.tf.Session() as sess:
        for image_data in get_batch(test_images,batch_size = 1):
            # batch size is set to 2 to distinguish between latency associated with first batch
            print(len(image_data))
            if len(image_data):
                print("进入查询")
                print ("Batch with {} images loaded in {} seconds".format(len(image_data),time.time()-start))
                start = time.time()
                count += 1
                features,files = inception.extract_features(image_data,sess)
                print ("Batch with {} images processed in {} seconds".format(len(features),time.time()-start))
                start = time.time()
                inception.store_index(features,files,count,test_index)

def get_batch(path,batch_size = 1000):
    """
    Args:
        path: directory containing images
    Returns: Generator with dictionary  containing image_file_nameh : image_data, each with size =  BUCKET_SIZE
    """
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

if __name__ == '__main__':
    test()
    file=['C:/Users/11725/PycharmProjects/tensorflow/myimagese/indexfile.pyimages/train/0\\0_42151.jpg',
     'C:/Users/11725/PycharmProjects/tensorflow/myimagese/indexfile.pyimages/train/0\\0_18386.jpg']
    filename=file[0]
    name=filename.split('\\')[1].split('_')
    wuti=name[0]
    if(wuti=='0'):
        print("这个可能是飞机")
    print(name)