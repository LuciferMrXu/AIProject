1. Ԥ����
   https://github.com/tzutalin/labelImg  ��ע����
   tf_convert_data.py �޸�pascalvoc_to_tfrecords.py 
   image_data = tf.gfile.FastGFile(filename, 'rb').read()
2. ѵ�� train_ssd_network.py
   ѵ��������tfrecordsĿ¼��
   ���model��checkpoints_newĿ¼�� 
   tf.app.flags.DEFINE_string(
    'dataset_name', 'pascalvoc_2007', 'The name of the dataset to load.')
    pascalvoc_000.tfrecord -->voc_2007_train_1.tfrecord
3. Ԥ��
   notebooks ssd_notebook.py
   notebooks video_use.py
   
   
���� ��
1. ���ܵ����̣�д���ʼǣ��پ���ϸ�ں��ص�
2. net.ssd_net SSDNet�� �ص����