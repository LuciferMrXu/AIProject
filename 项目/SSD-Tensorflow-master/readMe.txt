1. 预处理
   https://github.com/tzutalin/labelImg  标注工具
   tf_convert_data.py 修改pascalvoc_to_tfrecords.py 
   image_data = tf.gfile.FastGFile(filename, 'rb').read()
2. 训练 train_ssd_network.py
   训练数据在tfrecords目录下
   输出model在checkpoints_new目录下 
   tf.app.flags.DEFINE_string(
    'dataset_name', 'pascalvoc_2007', 'The name of the dataset to load.')
    pascalvoc_000.tfrecord -->voc_2007_train_1.tfrecord
3. 预测
   notebooks ssd_notebook.py
   notebooks video_use.py
   
   
讲解 ：
1. 先总的流程，写出笔记；再具体细节和重点
2. net.ssd_net SSDNet类 重点解释