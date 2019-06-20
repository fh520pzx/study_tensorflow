#coding:utf-8
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('./data/', one_hot=True)

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,W)+b)

#为了计算交叉熵，我们首先需要添加一个新的占位符用于输入正确值
y_ = tf.placeholder("float",[None,10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))  #计算目标类别与预测类别的交叉熵

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})

correct_predicition = tf.equal(tf.math.argmax(y,1),tf.math.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_predicition,"float"))
print(sess.run(accuracy,feed_dict={x:mnist.test.images , y_:mnist.test.labels}))

