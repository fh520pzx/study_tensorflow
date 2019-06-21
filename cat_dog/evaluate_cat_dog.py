import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import model_cat_dog
import os

def get_one_image(train):
    files = os.listdir(train)
    n = len(files)
    ind = np.random.randint(0,n)
    img_dir = os.path.join(train,files[ind])
    image = Image.open(img_dir)
    plt.imshow(image)
    plt.show()
    image = image.resize([208,208])
    image = np.array(image)
    return image

def evaluate_one_iamge():
    train = r'F:\cat_dog\test'
    image_array = get_one_image(train)

    with tf.Graph().as_default():
        BATCH_SIZE=1
        N_CLASSES=2
        image = tf.cast(image_array,tf.float32)
        image = tf.reshape(image,[1,208,208,3])
        logit = model_cat_dog.inference(image,BATCH_SIZE,N_CLASSES)
        logit = tf.nn.softmax(logit)

        x = tf.placeholder(tf.float32,shape=[208,208,3])
        log_train_dir = r'F:\cat_dog\train\savenet'

        saver = tf.train.Saver()
        with tf.Session() as sess:
            print("从指定路径中加载模型......")
            ckpt = tf.train.get_checkpoint_state(log_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print(ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('模型加载成功, 训练的步数为 %s' % global_step)
            else:
                print('模型加载失败，，，文件没有找到')

            prediction = sess.run(logit, feed_dict={x: image_array})
            # 获取输出结果中最大概率的索引
            max_index = np.argmax(prediction)
            if max_index == 0:
                print('猫的概率 %.6f' % prediction[:, 0])
            else:
                print('狗的概率 %.6f' % prediction[:, 1])
                # 测试
evaluate_one_iamge()
