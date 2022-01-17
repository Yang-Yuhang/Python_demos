import os
import glob
import cv2
import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

path = '/home/yyh/yyh_ws/src/人脸识别/2019_Faces_Good'  # 图片读取路径
w = 128  # 图片宽度
h = 128  # 图片高度
c = 3  # 图片通道数目

def read_img(path):
    imgs = []  # 图片
    labels = []  # 学号标签
    all_stu = []  # 所有学号的文件夹
    stu_num = []  # 学号(文件夹名称)

    for i in os.listdir(path):
        if os.path.isdir(path + '/' + i):
            stu_num.append(i)
            all_stu.append(path + '/' + i)

    # enumerate(a)遍历列表a,同时列出数据和下标（下标从零开始，下标在前数据在后)
    for id_num, folder in enumerate(all_stu):
        print(folder)
        print(id_num)
        for i in stu_num:
            if str(folder) == str(path + '/' + i):
                print(i)  # 对应学号，以便后续输出编号
        for im in glob.glob(folder + '/*.png'):  # 筛选文件夹下所有png图片（返回列表）
            img = cv2.imread(im)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换成RGB 形式
            img = cv2.resize(img, (w, h))
            imgs.append(img)
            labels.append(id_num)  # 标签为下标
    imgs = np.array(imgs)
    labels = np.array(labels)
    return imgs, labels


# 构建卷积神经网络
def CNNlayer():

    # 第一层卷积和池化
    # 卷积,128*128*3转为128*128*32
    conv1 = tf.layers.conv2d(inputs=x,    # 输入张量
                             filters=32,   # 卷积核数量
                             kernel_size=[5, 5],   # 卷积核大小
                             padding="same",  # same表示不够卷积核大小的块就补0,5*5卷积核补两个
                             activation=tf.nn.relu,   # 计算激活函数
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))  # 卷积核初始化:从截断的正态分布中输出随机值,stddev为标准偏差
    # 池化,128*128*32转为64*64*32
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # 第二层卷积和池化
    # 卷积,64*64*32转为64*64*64
    conv2 = tf.layers.conv2d(inputs=pool1,
                             filters=64,
                             kernel_size=[5, 5],
                             padding="same",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    # 池化,64*64*64转为32*32*64
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # 第三层卷积和池化
    # 卷积,32*32*64转为32*32*128
    conv3 = tf.layers.conv2d(inputs=pool2,
                             filters=128,
                             kernel_size=[3, 3],
                             padding="same",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    # 池化,32*32*128转为16*16*128
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    # 第四层卷积和池化
    # 卷积,16*16*128转为16*16*128
    conv4 = tf.layers.conv2d(inputs=pool3,
                             filters=128,
                             kernel_size=[3, 3],
                             padding="same",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    # 池化,16*16*128转为8*8*128
    pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

    # 池化输出变形,将图片拍扁为一维,大小为8192
    re1 = tf.reshape(pool4, [-1, 8 * 8 * 128])

    # 全连接层
    # 输出1024个接口
    dense1 = tf.layers.dense(inputs=re1,
                             units=1024,
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    # 输出512个接口
    dense2 = tf.layers.dense(inputs=dense1,
                             units=512,
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))

    # 输出层,输出100个接口
    put_out = tf.layers.dense(inputs=dense2,
                              units=100,
                              activation=None,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    return put_out


# 按批次取图片数据,参数分别为:图片数目,图片标签,一次输入的数据量,是否随机分
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_id in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            expected = indices[start_id:start_id + batch_size]
        else:
            expected = slice(start_id, start_id + batch_size)
        yield inputs[expected], targets[expected]


# 定义训练函数
def train():
    # 引入网络
    logits = CNNlayer()

    # 损失函数
    loss = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=logits)
    # 采用Adam优化器
    train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    # 计算正确率
    correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 训练和测试数据
    saver = tf.train.Saver(max_to_keep=3)  # 保存最优的３个模型
    max_acc = 0
    files = open('/home/yyh/yyh_ws/src/人脸识别/face_model/acc.txt', 'w')

    n_epoch = 20  # 训练次数
    batch_size = 64  # 每次输入64张图片
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epoch):
        # training
        train_loss, train_acc, n_batch = 0, 0, 0
        for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):

            x_train_a = x_train_a.astype('float32') / 255.0
            y_train_a = y_train_a.astype('int32')

            _, err, ac = sess.run([train_op, loss, acc], feed_dict={x: x_train_a, y_: y_train_a})
            train_loss += err
            train_acc += ac
            n_batch += 1
        print("train loss: %f" % (train_loss / n_batch))
        print("train acc: %f" % (train_acc / n_batch))

        # validation
        val_loss, val_acc, n_batch = 0, 0, 0
        for x_val_a, y_val_a in minibatches(x_test, y_test, batch_size, shuffle=False):

            x_val_a = x_val_a.astype('float32') / 255.0
            y_val_a = y_val_a.astype('int32')

            err, ac = sess.run([loss, acc], feed_dict={x: x_val_a, y_: y_val_a})
            val_loss += err
            val_acc += ac
            n_batch += 1
        print("validation loss: %f" % (val_loss / n_batch))
        print("validation acc: %f" % (val_acc / n_batch))
        files.write(str(epoch + 1) + ', val_acc: ' + str(val_acc) + '\n')
        if val_acc > max_acc:
            max_acc = val_acc
            saver.save(sess, '/home/yyh/yyh_ws/src/人脸识别/face_model/face_model.ckpt', global_step=epoch + 1)
    files.close()
    sess.close()


if __name__ == '__main__':

    # 占位符
    x = tf.compat.v1.placeholder(tf.float32, shape=[None, w, h, c], name='x')
    y_ = tf.compat.v1.placeholder(tf.int32, shape=[None, ], name='y_')

    # 读取数据集
    data, labs = read_img(path)
    print('图片总数:', data.shape[0])  # 图片总数
    print('标签总数:', labs.shape[0])  # 标签总数

    # 将数据分为训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(data, labs, test_size=0.2, random_state=random.randint(0, 100))

    # 开始训练
    train()
