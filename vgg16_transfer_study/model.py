import tensorflow as tf
import pandas as pd
import numpy as np
# import transfer_learning
from sklearn.preprocessing import LabelBinarizer

codes = np.load('codes.npy')
labels = np.load('labels.npy')
lb = LabelBinarizer()
lb.fit(labels)
labels_ves = lb.transform(labels)
label = []
for i in labels_ves:
    label.append(i[0])
# one_hot=tf.one_hot(label,2)

def get_batch(codes,label):
    input_queue = tf.train.slice_input_producer([codes, label])
    codes_batch,lable_batch = tf.train.batch(input_queue,batch_size=batch_size,num_threads=10,capacity=64) #num_threads=10,
    lable_batch = tf.reshape(lable_batch,[batch_size])
    return codes_batch,lable_batch

def get_logits(images,n_class):
    with tf.variable_scope('logits') as scope:
        fc = tf.contrib.layers.fully_connected(images,256)
        logits = tf.contrib.layers.fully_connected(fc,n_class)
        tf.summary.scalar(scope.name + '/logits', logits)
    return logits

def loss(logits,labels):
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.reduce_mean(cross_entropy)
        tf.summary.scalar(scope.name + '/loss', loss)
    return loss

def training(cost,learning_rate):
    with tf.name_scope('optimiazer'):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_step = optimizer.minimize(cost)
    return train_step

def accuracy(logits,labels):
    with tf.variable_scope('accuracy') as scope:
        # predicted = tf.nn.softmax(logits)
        # correct_pred = tf.equal(tf.argmax(predicted, 1), tf.argmax(labels,1))
        # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + '/accuracy', accuracy)
    return accuracy

batch_size=20
cod,lab = get_batch(codes,label)
# cod,lab = get_batch(codes,one_hot)
train_logits = get_logits(cod,2)
train_loss = loss(train_logits,lab)
train_op = training(train_loss,0.0001)
train_acc = accuracy(train_logits,lab)


with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)
    try:
        for i in range(50000):
            loss,acc = sess.run([train_loss,train_acc])
            if(i%50==0):
                print(i,loss,acc)
        saver.save(sess, "checkpoints/flowers.ckpt")
    except tf.errors.OutOfRangeError:
        print('done')
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()




