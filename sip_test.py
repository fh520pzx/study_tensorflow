import tensorflow as tf
import numpy as np

# images = ['img1','img2','img3','img4','img5']
# lables = [1,2,3,4,5]
# epoch_num=10
#
# f= tf.train.slice_input_producer([images,lables],num_epochs=None,shuffle=False)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess=sess,coord=coord)
#     for i in range(epoch_num):
#         k = sess.run(f)
#         print(i,k)
#     coord.request_stop()
#     coord.join(threads)

#样本数量
sample_num = 5
#迭代次数
epoch_num = 2
#每次的训练大小
batch_size = 2
#一次迭代中的batch个数
batch_total = int(sample_num/batch_size)+1

#生产数据以及标签
def generate_data(sample_num=sample_num):
    lables = np.asarray(range(0,sample_num))
    images = np.random.random([sample_num,224,224,3])
    print(images.shape,lables.shape)
    return images,lables

def get_batch_data(batch_size=batch_size):
    images,lable = generate_data()
    images = tf.cast(images,tf.float32)
    lable = tf.cast(lable,tf.int32)
    #从tensor列表中按顺序或者随机抽取一个tensor
    input_queue = tf.train.slice_input_producer([images,lable],shuffle=True)
    image_batch,lable_batch = tf.train.batch(input_queue,batch_size=batch_size,num_threads=1,capacity=64)
    return image_batch,lable_batch

image_batch,label_batch = get_batch_data(batch_size=batch_size)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess,coord)
    try:
        for i in range(epoch_num):
            print('*******************')
            for j in range(batch_total):
                image_batch_v,label_batch_v = sess.run([image_batch,label_batch])
                print(image_batch_v.shape,label_batch_v)
    except tf.errors.OutOfRangeError:
        print('done')
    finally:
        coord.request_stop()
    coord.join(threads)