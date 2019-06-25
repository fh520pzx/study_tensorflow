import tensorflow as tf
import numpy as np
import os
from tensorflow_vgg import vgg16
from tensorflow_vgg import utils


def create_data():
    data_dir = r'flower_photos/'
    contents = os.listdir(data_dir)
    classes = [each for each in contents if os.path.isdir(data_dir + each)]

    batch_size = 10
    codes_list = []  # 存放特征值
    labels = []  # 存放花的类别
    batch = []  # 存放图片数据
    codes = None

    with tf.Session() as sess:
        vgg = vgg16.Vgg16()
        input_ = tf.placeholder(tf.float32,[None,224,224,3])
        with tf.name_scope('content_vgg'):
            vgg.build(input_)

        #为每个不同类别的花分别用vgg16计算特征值
        for each in classes:
            print("starting {} images".format(each))
            class_path = data_dir+each
            files = os.listdir(class_path)
            #枚举每种花的文件夹中图片
            for ii,file in enumerate(files,1):
                img = utils.load_image(os.path.join(class_path,file))
                batch.append(np.reshape(img,(1,224,224,3)))
                labels.append(each)

                if ii%batch_size==0 or ii==len(files):
                    images = np.concatenate(batch)   #s数组拼接
                    feed_dict = {input_:images}

                    #计算特征值
                    codes_batch = sess.run(vgg.relu6,feed_dict=feed_dict)         #得到第一层全连接层之后的特征值

                    #将结果加入到codes数组中
                    if codes is None:
                        codes = codes_batch
                    else:
                        codes = np.concatenate((codes,codes_batch))
                    batch=[]
                    print('{} images processed'.format(ii))
    return codes,labels






