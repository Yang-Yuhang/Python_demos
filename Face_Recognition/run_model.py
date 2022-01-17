import cv2
import dlib
import sys
import tensorflow as tf

from skimage import transform


# 定义相关参数
w = 128
h = 128
c = 3
# 占位符
x = tf.compat.v1.placeholder(tf.float32,shape=[None, w, h, c], name='x')
y_ = tf.compat.v1.placeholder(tf.int32,shape=[None, ], name='y_')

ID = (1711519, 1711496, 1711472, 1711528, 1711512, 1711505, 1711460, 1711465, 1711488,
      1711469, 1711479, 1711495, 1711476, 1711514, 1711452, 1711485, 1711509, 1711533,
      1711515, 1711517, 1711520, 1711494, 1711454, 1711448, 1711467, 1711461, 1711464,
      1710402, 1710696, 1711524, 1711459, 1711518, 1711511, 1711530, 1711455, 1711504,
      1711503, 1711491, 1711492, 1511293, 1711525, 1711497, 1713402, 1711475, 1711502,
      1711521, 1711487, 1711501, 1711480, 1711523, 1711500, 1711470, 1711484, 1711471,
      1711462, 1711450, 1711474, 1711536, 1711535, 1711458, 1711506, 1710675, 1711486,
      1711457, 1711522, 1711516, 1711499, 1711466, 1711529, 1711456, 1711510, 1711449,
      1711481, 1711489, 1711508, 1711527, 1711468, 1711482, 1711507, 1711453, 1711463,
      1711498, 1711477, 1711526)


def CNNlayer():
    # 第一层卷积和池化
    # 卷积
    conv1 = tf.layers.conv2d(
        inputs=x,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    # 池化层
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # 第二层卷积和池化
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # 第三层卷积和池化
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=128,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    # 第四层卷积和池化
    conv4 = tf.layers.conv2d(
        inputs=pool3,
        filters=128,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)
    # 池化输出变形
    re1 = tf.reshape(pool4, [-1, 8 * 8 * 128])
    # 全连接层
    dense1 = tf.layers.dense(inputs=re1,
                             units=1024,
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    dense2 = tf.layers.dense(inputs=dense1,
                             units=512,
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    put_out = tf.layers.dense(inputs=dense2,
                              units=100,
                              activation=None,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    return put_out


# 识别
if __name__ == '__main__':
    # 预测
    put_out = CNNlayer()
    predict = tf.argmax(put_out, 1)
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, '/home/yyh/yyh_ws/src/人脸识别/face_model/face_model.ckpt-17')  # 选择模型

    # 检测人脸
    detector = dlib.get_frontal_face_detector()  # 载入特征提取器

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    while True:
        r, frame = cap.read()
        cv2.imshow('frame', frame)
        # 抓取图像:按下数字１
        if cv2.waitKey(30) == 49:
            cv2.imwrite('/home/yyh/yyh_ws/src/人脸识别/now.png', frame)
            img = cv2.imread("/home/yyh/yyh_ws/src/人脸识别/now.png")
            dets = detector(img, 1)
            # 在图片中只检测到一个人脸时操作
            if len(dets) == 1:
                for index, face in enumerate(dets):
                    top = face.top()
                    bottom = face.bottom()
                    right = face.right()
                    left = face.left()

                    cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 3)
                    cv2.imshow('rect', img)

                    test_img = img[top:bottom, left:right]  # 定点
                    test_img = transform.resize(test_img, (w, h, c))  # 裁剪
                    cv2.imshow('test_img', test_img)

                    res = sess.run(predict, feed_dict={x: [test_img]})  # 进行识别
                    print('检测编号为:', res[0])  # 显示结果
                    print('最终检测结果为:', ID[res[0]])

            else:
                print('人脸过多，请重新拍摄。')
            f = cv2.waitKey(10000)  # 等待10秒
            if f == 27:
                break
        # 直接退出
        if cv2.waitKey(30) == 27:
            cap.release()
            cv2.destroyAllWindows()
            sys.exit()
    cap.release()
    cv2.destroyAllWindows()

