import os,sys,logging,time,shutil
import inception


USER = ""
HOST = ""
AWS = sys.platform != ''

private_key =  ""
CONFIG_PATH = __file__.split('settings.py')[0]
INDEX_PATH = CONFIG_PATH+"index/"
DATA_PATH = CONFIG_PATH+"images"

def index():
    """
    Index images
    """
    logging.info("Starting with images present in {} storing index in {}".format(DATA_PATH,INDEX_PATH))
    if (os.path.exists(INDEX_PATH)):
        inception.load_network()
        print("加载模型成功")
        count = 30
        start = time.time()
        with inception.tf.Session() as sess:
            #DATA_PATH是要图片路径
            for image_data in inception.get_batch(DATA_PATH):
                logging.info("Batch with {} images loaded in {} seconds".format(len(image_data), time.time() - start))
                start = time.time()
                count += 1
                #抽取特征
                features, files = inception.extract_features(image_data, sess)
                logging.info("Batch with {} images processed in {} seconds".format(len(features), time.time() - start))
                start = time.time()
                #存储我们获取到的特征
                inception.store_index(features, files, count, INDEX_PATH)
    else:
        try:
            os.mkdir(INDEX_PATH)
        except:
            print ("Could not created {}, if its on /mnt/ have you set correct permissions?".format(INDEX_PATH))
            raise ValueError

if __name__ == '__main__':
    index()

