import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

learning_rate = 0.01
training_epochs = 2000
display_step = 50

train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = train_X.shape[0]
print(n_samples)
X = tf.placeholder("float")
Y = tf.placeholder("float")

W = tf.Variable(np.random.randn(),name="weight")
b = tf.Variable(np.random.randn(),name="bais")

activation = tf.add(tf.multiply(X,W),b)

#损失函数
cost = tf.reduce_sum(tf.pow(activation-Y,2))/(2*n_samples)

#梯度下降
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epochs in range(training_epochs):
        for x,y in zip(train_X,train_Y):
            sess.run(optimizer,feed_dict={X:x,Y:y})
        if (epochs%display_step == 0):
            print(epochs,sess.run(cost,feed_dict={X: train_X, Y:train_Y}),sess.run(W),sess.run(b))

    plt.plot(train_X,train_Y,'ro')
    plt.plot(train_X,sess.run(W)*train_X+sess.run(b))
    plt.show()