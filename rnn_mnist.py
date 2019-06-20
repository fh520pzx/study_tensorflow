from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

train_rate = 0.001
train_step = 10000
batch_size = 1280
display_step = 100
frame_size = 28  #序列里面每个分量的大小
sequence_size = 28 #序列的长度
hidden_num = 100 #隐藏层层数
n_class = 10 #列别种类

mnist = input_data.read_data_sets('./data/', one_hot=True)
x = tf.placeholder("float",[None,sequence_size*frame_size],name="inputx")
y_ = tf.placeholder("float",[None,n_class],name="expected_y")

weight = tf.Variable(tf.truncated_normal(shape=[hidden_num,n_class]))
bais = tf.Variable(tf.zeros(shape=[n_class]))

def RNN(x,weight,bais):
    # 先把输入转换成为dynamic_rnn接受的形状：batch,sequence_length,frame_size这样子
    x = tf.reshape(x,shape=[-1,sequence_size,frame_size])
    #生成hidden_num个隐层的RNN网络，rnn_cell.output_size等于隐层个数
    rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_num)
    init_state = tf.zeros(shape=[batch_size, rnn_cell.state_size])
    output,states = tf.nn.dynamic_rnn(rnn_cell,x,dtype="float")
    #此时output就是一个[batch_size,sequence_length,rnn_cell.output_size]形状的tensor
    return tf.nn.softmax(tf.matmul(output[:,-1,:],weight)+bais,1)
    #output[:,-1,:]形状为[batch_size,rnn_cell.output_size]，也就是：[batch_size,hidden_num]
predy=RNN(x,weight,bais)
# cross_entropy = -tf.reduce_sum(y_*tf.log(RNN(x,weight,bais)))   #计算目标类别与预测类别的交叉熵
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predy,labels=y_))
train_step = tf.train.AdagradOptimizer(train_rate).minimize(cross_entropy)  #梯度下降



correct_prediction = tf.equal(tf.argmax(predy, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        batch = mnist.train.next_batch(batch_size)
        if (i%50==0):
            print(i,sess.run(accuracy,feed_dict={x: batch[0], y_: batch[1]}))
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})
    print(sess.run(accuracy,feed_dict={x:mnist.test.images , y_:mnist.test.labels}))
