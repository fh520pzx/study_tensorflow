import tensorflow as tf
import pandas as pd
import numpy as np
import transfer_learning
from sklearn.preprocessing import LabelBinarizer

codes,labels = transfer_learning.create_data()
lb = LabelBinarizer()
lb.fit(labels)
labels_ves = lb.transform(labels)
print(labels_ves)


input_ = tf.placeholder(tf.float32,[None,codes.shape[1]])
labels_ = tf.placeholder(tf.int64,[None,labels_ves.shape[1]])
#添加一层256维的全连接层
fc = tf.contrib.layers.fully_connected(input_,256)
#添加一层5维的全连接层
logits = tf.contrib.layers.fully_connected(fc,3)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels_)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(0.0001)
train_step = optimizer.minimize(cost)
predicted = tf.nn.softmax(logits)

#计算精确度
correct_pred = tf.equal(tf.argmax(predicted,1),tf.argmax(labels_,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))


def get_batch(codes,label):
    input_queue = tf.train.slice_input_producer([codes, label])
    codes_batch,lable_batch = tf.train.batch(input_queue,batch_size=batch_size,num_threads=10,capacity=64) #num_threads=10,
    lable_batch = tf.reshape(lable_batch,[batch_size])
    return codes_batch,lable_batch

batch_size=20
with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)
    cod,lab = get_batch(codes,labels_ves)
    try:
        for i in range(1000):
            cod,lab = sess.run([cod,lab])
            feed = {input_:cod,labels_:lab}
            loss,_ = sess.run([cost,optimizer],feed_dict=feed)
            acc = sess.run(accuracy,feed_dict=feed)
            if(i%50==0):
                print(i,loss,accuracy)
        saver.save(sess, "checkpoints/flowers.ckpt")
    except tf.errors.OutOfRangeError:
        print('done')
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()




