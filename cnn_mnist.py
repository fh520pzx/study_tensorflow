from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


mnist = input_data.read_data_sets('./data/', one_hot=True)
x = tf.placeholder("float",[None,784])
y_ = tf.placeholder("float",[10])

def weight_varibale(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bais_varibale(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

#卷积层
def conv2d(x,W):
    """
    :param x: shape:[图片数量，高度，宽度，图片通道数]
    :param W: 卷积核：[高度，宽度，图片通道数，卷积核数量]
    :return:
    """
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")
#池化层
def max_pol_2x2(x):
    """
    :param x: 特征图[数量，高度，宽度，通道数]
    :return:
    """
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

#第一层卷积+池化
w_conv1 = weight_varibale([5,5,1,32])
b_conv1 = bais_varibale([32])

x_image = tf.reshape(x,[-1,28,28,1])
h_conv1 = tf.nn.relu(conv2d(x_image,w_conv1)+b_conv1)
h_pool1 = max_pol_2x2(h_conv1)

#第二层卷积+池化
w_conv2 = weight_varibale([5,5,32,64])
b_conv2 = bais_varibale([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2)
h_pool2 = max_pol_2x2(h_conv2)

#全连接层
w_fc1 = weight_varibale([7*7*64,1024])
b_fc1 = bais_varibale([1024])

h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

w_fc2 = weight_varibale([1024,10])
b_fc2 = bais_varibale([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2)+b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))   #计算目标类别与预测类别的交叉熵
train_step = tf.train.AdagradOptimizer(1e-4).minimize(cross_entropy)  #梯度下降

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        batch = mnist.train.next_batch(100)
        if (i%50==0):
            print(i,sess.run(accuracy,feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0}))
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1],keep_prob:0.5})
    print(sess.run(accuracy,feed_dict={x:mnist.test.images , y_:mnist.test.labels,keep_prob:1.0}))