import tensorflow as tf
import numpy as np
import os

def get_files(file_dir):
    cats = []
    lable_cats = []
    dogs = []
    lable_dogs = []
    #cat-0 dog-1
    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        if 'cat' in name[0]:
            cats.append(file_dir+'/'+file)
            lable_cats.append(0)
        else:
            if 'dog' in name[0]:
                dogs.append(file_dir+'/'+file)
                lable_dogs.append(1)
        image_list = np.hstack((cats,dogs))
        lable_list = np.hstack((lable_cats,lable_dogs))
    temp = np.array([image_list,lable_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    image_list = temp[:,0]
    lable_list = temp[:,1]
    lable_list = [int(i) for i in lable_list]
    return image_list,lable_list

def get_batch(image,label,image_W,image_H,batch_size,capacity):
    image = tf.cast(image,tf.string)
    label = tf.cast(label,tf.int32)

    input_queue = tf.train.slice_input_producer([image, label])
    label = input_queue[1]

    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents,channels=3)
    image = tf.image.resize_image_with_crop_or_pad(image,image_W,image_H)
    image = tf.image.per_image_standardization(image)

    image_batch,lable_batch = tf.train.batch([image,label],batch_size=batch_size,num_threads=10,capacity=capacity) #num_threads=10,
    lable_batch = tf.reshape(lable_batch,[batch_size])
    image_batch = tf.cast(image_batch,tf.float32)
    return image_batch,lable_batch


def inference(images, batch_size, n_classes):
    with tf.variable_scope('conv1') as scope:
        # 卷积盒的为 3*3 的卷积盒，图片厚度是3，输出是16个featuremap
        weights = tf.get_variable('weights',
                                  shape=[3, 3, 3, 16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pooling1')
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3, 3, 16, 16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name='conv2')

        # pool2 and norm2
    with tf.variable_scope('pooling2_lrn') as scope:
        norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='pooling2')

    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool2, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights',
                                  shape=[dim, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    # local4
    with tf.variable_scope('local4') as scope:
        weights = tf.get_variable('weights',
                                  shape=[128, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')

        # softmax
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('softmax_linear',
                                  shape=[128, n_classes],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[n_classes],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')

    return softmax_linear


def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits \
            (logits=logits, labels=labels, name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name + '/loss', loss)
    return loss


def trainning(loss, learning_rate):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + '/accuracy', accuracy)
    return accuracy

n_classses=2
image_w = 208
image_h = 208
batch_size = 32
capacity = 256
max_step = 10000
learning_rate = 0.0001

def run_training():
    train_dir = r'F:\cat_dog\train'
    log_train_dir = r'F:\cat_dog\train\savenet'
    train,train_label = get_files(train_dir)
    train_batch,train_label_batch = get_batch(train,train_label,image_w,image_h,batch_size,capacity)

    train_logits = inference(train_batch,batch_size,n_classses)
    train_loss = losses(train_logits,train_label_batch)
    train_op = trainning(train_loss,learning_rate)
    train_acc = evaluation(train_logits,train_label_batch)

    # 合并 summary
    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    # 保存summary
    train_writer = tf.summary.FileWriter(log_train_dir, sess.graph)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)
    try:
        for i in range(max_step):
            if coord.should_stop():
                break
            _,tra_loss,tra_acc = sess.run([train_op,train_loss,train_acc])
            if(i%50==0):
                print(i,tra_loss,tra_acc)
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, i)

            if i % 2000 == 0 or (i + 1) == max_step:
                # 每隔2000步保存一下模型，模型保存在 checkpoint_path 中
                checkpoint_path = os.path.join(log_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=i)
    except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()
run_training()

