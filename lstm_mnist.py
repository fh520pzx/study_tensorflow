from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


time_steps = 28
num_hidden = 128
n_input = 28
learning_rate = 0.001
n_classes = 10
batch_size = 1280

mnist = input_data.read_data_sets('./data/',one_hot=True)
x = tf.placeholder("float",[None,time_steps*n_input],name="inputx")
y_ = tf.placeholder("float",[None,n_classes],name="expectedy")

weights = tf.Variable(tf.random_normal([num_hidden,n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))

input = tf.reshape(x,[-1,time_steps,n_input])

lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden)
output,_ = tf.nn.dynamic_rnn(lstm_cell,input,dtype="float")
prediction = tf.matmul(output[:,-1,:],weights)+bias

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y_))
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(cross_entropy)  #梯度下降

correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        batch = mnist.train.next_batch(batch_size)
        if (i%50==0):
            print(i,sess.run(accuracy,feed_dict={x: batch[0], y_: batch[1]}))
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})
    print(sess.run(accuracy,feed_dict={x:mnist.test.images , y_:mnist.test.labels}))
