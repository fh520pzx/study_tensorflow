import tensorflow as tf
import numpy as np
import os
import cv2

image_w = 600
image_h = 600

Ratio = None
INI_NOISE_RATIO = 0.7
STYLE_STRENGTH = 500
ITERATION = 2000


content_img = 'content_img.jpg'
style_img = 'style_img.jpg'

content_layers = [('conv4_3',1.)]
style_layers = [('conv1_1',2.),('conv2_1',1),('conv3_1',0.5),('conv4_1',0.25),('conv5_1',0.125)]

layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3',
          'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3']

#初始化vgg16
def vgg16(input,model_path=None):
    if (model_path is None):
        model_path = r'D:\MyData\fanghui3\PycharmProjects\Test\tf_study\transfer_vgg_16\tensorflow_vgg\vgg16.npy'

    if (os.path.isfile(model_path)==False):
        raise FileNotFoundError('vgg16.npy cannot be found!!!')

    #遍历其内建值对，导入模型参数，其实保存的就是各个参数
    wDict = np.load(model_path, encoding="latin1").item()
    net = {}
    net['input'] = input
    # conv1_1
    weight1_1 = tf.Variable(wDict['conv1_1'][0], trainable=False)
    bias1_1 = tf.Variable(wDict['conv1_1'][1], trainable=False)
    net['conv1_1'] = tf.nn.relu(tf.nn.conv2d(net['input'], weight1_1, [1, 1, 1, 1], 'SAME') + bias1_1)

    # conv1_2
    weight1_2 = tf.Variable(wDict['conv1_2'][0], trainable=False)
    bias1_2 = tf.Variable(wDict['conv1_2'][1], trainable=False)
    net['conv1_2'] = tf.nn.relu(tf.nn.conv2d(net['conv1_1'], weight1_2, [1, 1, 1, 1], 'SAME') + bias1_2)

    # pool1
    net['pool1'] = tf.nn.avg_pool(net['conv1_2'], [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

    # conv2_1
    weight2_1 = tf.Variable(wDict['conv2_1'][0], trainable=False)
    bias2_1 = tf.Variable(wDict['conv2_2'][1], trainable=False)
    net['conv2_1'] = tf.nn.relu(tf.nn.conv2d(net['pool1'], weight2_1, [1, 1, 1, 1], 'SAME') + bias2_1)

    # conv2_2
    weight2_2 = tf.Variable(wDict['conv2_2'][0], trainable=False)
    bias2_2 = tf.Variable(wDict['conv2_2'][1], trainable=False)
    net['conv2_2'] = tf.nn.relu(tf.nn.conv2d(net['conv2_1'], weight2_2, [1, 1, 1, 1], 'SAME') + bias2_2)

    # pool2
    net['pool2'] = tf.nn.avg_pool(net['conv2_2'], [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

    # conv3_1
    weight3_1 = tf.Variable(wDict['conv3_1'][0], trainable=False)
    bias3_1 = tf.Variable(wDict['conv3_1'][1], trainable=False)
    net['conv3_1'] = tf.nn.relu(tf.nn.conv2d(net['pool2'], weight3_1, [1, 1, 1, 1], 'SAME') + bias3_1)

    # conv3_2
    weight3_2 = tf.Variable(wDict['conv3_2'][0], trainable=False)
    bias3_2 = tf.Variable(wDict['conv3_2'][1], trainable=False)
    net['conv3_2'] = tf.nn.relu(tf.nn.conv2d(net['conv3_1'], weight3_2, [1, 1, 1, 1], 'SAME') + bias3_2)

    # conv3_3
    weight3_3 = tf.Variable(wDict['conv3_3'][0], trainable=False)
    bias3_3 = tf.Variable(wDict['conv3_3'][1], trainable=False)
    net['conv3_3'] = tf.nn.relu(tf.nn.conv2d(net['conv3_2'], weight3_3, [1, 1, 1, 1], 'SAME') + bias3_3)


    # pool3
    net['pool3'] = tf.nn.avg_pool(net['conv3_3'], [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

    # conv4_1
    weight4_1 = tf.Variable(wDict['conv4_1'][0], trainable=False)
    bias4_1 = tf.Variable(wDict['conv4_1'][1], trainable=False)
    net['conv4_1'] = tf.nn.relu(tf.nn.conv2d(net['pool3'], weight4_1, [1, 1, 1, 1], 'SAME') + bias4_1)

    # conv4_2
    weight4_2 = tf.Variable(wDict['conv4_2'][0], trainable=False)
    bias4_2 = tf.Variable(wDict['conv4_2'][1], trainable=False)
    net['conv4_2'] = tf.nn.relu(tf.nn.conv2d(net['conv4_1'], weight4_2, [1, 1, 1, 1], 'SAME') + bias4_2)

    # conv4_3
    weight4_3 = tf.Variable(wDict['conv4_3'][0], trainable=False)
    bias4_3 = tf.Variable(wDict['conv4_3'][1], trainable=False)
    net['conv4_3'] = tf.nn.relu(tf.nn.conv2d(net['conv4_2'], weight4_3, [1, 1, 1, 1], 'SAME') + bias4_3)

    # pool4
    net['pool4'] = tf.nn.avg_pool(net['conv4_3'], [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

    # conv5_1
    weight5_1 = tf.Variable(wDict['conv5_1'][0], trainable=False)
    bias5_1 = tf.Variable(wDict['conv5_1'][1], trainable=False)
    net['conv5_1'] = tf.nn.relu(tf.nn.conv2d(net['pool4'], weight5_1, [1, 1, 1, 1], 'SAME') + bias5_1)

    # conv5_2
    weight5_2 = tf.Variable(wDict['conv5_2'][0], trainable=False)
    bias5_2 = tf.Variable(wDict['conv5_2'][1], trainable=False)
    net['conv5_2'] = tf.nn.relu(tf.nn.conv2d(net['conv5_1'], weight5_2, [1, 1, 1, 1], 'SAME') + bias5_2)

    # conv5_3
    weight5_3 = tf.Variable(wDict['conv5_3'][0], trainable=False)
    bias5_3 = tf.Variable(wDict['conv5_3'][1], trainable=False)
    net['conv5_3'] = tf.nn.relu(tf.nn.conv2d(net['conv5_2'], weight5_3, [1, 1, 1, 1], 'SAME') + bias5_3)

    # pool5
    net['pool5'] = tf.nn.avg_pool(net['conv5_3'], [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    return net

#计算gram矩阵（向量内积所组成的矩阵）
def gram_matrix(tensor, length, depth):
    tensor = tf.reshape(tensor, (length, depth))
    return tf.matmul(tf.transpose(tensor), tensor)

#计算白噪声图片与内容图像之间的损失函数
#比较生成的图片和内容图片在训练好的网络中某一层的特征图的相似度（计算两个特征图的l2距离）
def build_content_loss(combination, content):
    content_sum = 0.0
    for i, j in enumerate(content_layers):
        #i对应的为个数，j对应的为元素
        shape = combination[j[0]].get_shape() #得到在vgg16中该层的形状[个数，高度，宽度，通道数/特征个数]
        M = shape[1].value * shape[2].value
        N = shape[3].value
        content_sum += j[1] * 0.25/(M ** 0.5 + N ** 0.5) * tf.reduce_sum(tf.pow(combination[j[0]] - content[j[0]], 2))
    return content_sum

#计算白噪声图片与风格图片之间的损失函数
def build_style_loss(combination, style):
    style_sum = 0.0
    for i, j in enumerate(style_layers):
        shape = combination[j[0]].get_shape()
        M = shape[1].value * shape[2].value
        N = shape[3].value
        para1 = combination[j[0]]
        para2 = style[j[0]]
        sub = gram_matrix(para1, M, N) - gram_matrix(para2, M, N)
        sum = tf.reduce_sum(tf.pow(sub, 2))
        pre = j[1] * 1.0 / (4 * N ** 2 * M ** 2)
        style_sum += tf.multiply(pre, sum)
    return style_sum

def main():
    myinput = tf.placeholder(dtype=tf.float32, shape=[1, image_h, image_w, 3])

    raw_styleimg = cv2.imread(style_img)
    raw_styleimg = cv2.resize(raw_styleimg, (image_h, image_w))
    styleimg = np.expand_dims(raw_styleimg, 0)
    # 图片进行归一化
    styleimg[0][0] -= 123
    styleimg[0][1] -= 117
    styleimg[0][2] -= 104
    styleimg = tf.Variable(styleimg, dtype=tf.float32, trainable=False)

    raw_contentimg = cv2.imread(content_img)
    Ratio = raw_contentimg.shape #获得内容图片的长宽，方便最后还原图片的样式
    raw_contentimg = cv2.resize(raw_contentimg, (image_h, image_w))
    contentimg = np.expand_dims(raw_contentimg, 0)
    contentimg[0][0] -= 123
    contentimg[0][1] -= 117
    contentimg[0][2] -= 104

    contentimg = tf.Variable(contentimg, dtype=tf.float32, trainable=False)

    #在内容图片的基础上获得一张白噪声图片
    combination = INI_NOISE_RATIO*np.random.uniform(-20, 20, (1, image_h, image_w, 3)).astype('float32') \
                  + \
                  (1.-INI_NOISE_RATIO) * contentimg
    combination = tf.Variable(combination, dtype=tf.float32, trainable=True)

    #相乘是为了变化矩阵的形状
    stylenet = vgg16( myinput *styleimg)
    contentnet = vgg16( myinput *contentimg)
    combinationnet = vgg16(myinput * combination)      #白噪声图片网络


    loss = 500 * build_style_loss(combinationnet, stylenet) + build_content_loss(combinationnet, contentnet)
    train = tf.train.AdamOptimizer(2).minimize(loss)

    img = np.ones(dtype=np.float32, shape=[1, image_h, image_w, 3])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(ITERATION):
            print(sess.run(loss, feed_dict={myinput: img}))
            sess.run(train, feed_dict={myinput: img})
            pic = sess.run(combination, feed_dict={myinput: img})[0]
            pic[0] += 123
            pic[1] += 117
            pic[2] += 104
            if(i%100==0):
                cv2.imwrite('result\%d.jpg' % i, cv2.resize(pic, (Ratio[1], Ratio[0])))

if __name__ == '__main__':
    main()

