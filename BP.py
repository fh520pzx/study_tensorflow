#  coding:utf8

import tensorflow as tf
from numpy.random import RandomState

#定义训练数据batch大小
batch_size = 8

#声明w1,w2两个变量,这里还通过seed参数设定随机种子,保证每次运行结果一致
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

#在shape的一个维度上使用None,方便使用不大的batch大小,训练时使用小的batch,测试时使用全部的数据(内存不溢出的前提下)
x = tf.placeholder(tf.float32, shape=(None, 2), name='x_input')
y_= tf.placeholder(tf.float32, shape=(None,1),name='y_input')

#定义数据网络前向传播过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

#定义损失函数和BP算法
cross_entropy = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

learning_rate = 0.001
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_step = optimizer.minimize(cross_entropy)

#通过随机数生成一个数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size,2)

#定义规则来给出样本标签,在这里所有x1+x2<1的样例都被认为是正样本, 0代表负样本 1代表正样本
Y = [[int(x1+x2<1)] for (x1,x2) in X]

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(w1))
print(sess.run(w2))
for i in range(500000):
    start = (i*batch_size)%dataset_size
    end = min(start+batch_size,dataset_size)
    sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
    if(i%1000==0):
        total_cross_entropy = sess.run(cross_entropy,feed_dict={x:X,y_:Y})
        print("After %d training setp(s),cross entropy on all data is %g"%(i,total_cross_entropy))
print(sess.run(w1))
print(sess.run(w2))

