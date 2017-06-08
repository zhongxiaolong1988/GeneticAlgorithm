# -*- coding: UTF-8 -*-

import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  #读取图片数据集
sess = tf.InteractiveSession()

#---------------------------重要函数声明-------------------------------------
#权重变量
def weight_variable(shape):
    #输出正态分布的随机值，标准差为0.1，默认最大为1，最小为-1，均值为0
    inital = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(inital)

#偏移变量
def bias_veriable(shape):
    #创建一个结构为shape的矩阵，或者说数组，声明其行和列，初始化所有值为0.1
    inital = tf.constant(0.1, shape=shape)
    return tf.Variable(inital)

#卷积函数，大概意思是根据步长一步一步去遍历输入的值，行成特征向量输出到下一层
def conv2d(x, W):
    #卷积遍历各个方向步数为1，SAME表示边缘自动补0而不是丢弃，遍历相乘
    #strides：表示步长：当输入的默认格式为：“NHWC”，
    #则 strides = [batch , in_height , in_width, in_channels]。
    #其中 batch 和 in_channels 要求一定为1，即只能在一个样本的一个通道上的特征图上进行移动，
    #in_height , in_width表示卷积核在特征图的高度和宽度上移动的布长
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#池化函数：意思是把输入的高宽变小，比如4*4 变成2* 2，有两种方式，一种是取每一个小块的最大值，另外一种是取每一个小块的平均值
def max_pool_2_2(x):
    #池化卷积结果(conv2d),池化层的kernal采用和2*2的大小，步数也为2，周围填0，取最大值，数量量缩小
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#--------------------------定义输入输出结构---------------------------------
#声明一个占位符，None表示输入的图片数量不定，28*28的图片分辨率
xs = tf.placeholder(tf.float32, [None, 784])
#输出类别总共是0-9个类别，对应分类的输出结果
ys = tf.placeholder(tf.float32, [None, 10])

#x_image又把xs reshape成 28*28*1的形状，因为是灰色图片，所以通道是1，作为训练时的input，-1代表图片数量不定
x_image = tf.reshape(xs, [-1, 28, 28, 1])

#-------------------------搭建网络，定义算法公式，也就是前进时候具体应该怎么计算----------

#第一层卷积操作
#第一次卷积核的大小,即patch,第三个参数是图像通道数，第四个参数是卷积核数目，代表会出现多少个卷积特征图像
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_veriable([32]) #偏移b的数量应该和w一致
#图片乘以卷积核加上偏移量，卷积结果为28 * 28 * 32
#relu函数表示一个激活函数，在大于0时激活，激活到第一象限，其他的收敛于一个接近于0的小数
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

#结果池化，变成14*14*32的池化结果作为下一层的输入
h_pool1 = max_pool_2_2(h_conv1)

#第二次卷积
#32个通道卷积出64个特征
w_conv2 = weight_variable([5, 5, 32, 64])
#64个偏移数据
b_conv2 = bias_veriable([64])
#注意上一层池化的结果作为输入
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
#池化结果
h_pool2 = max_pool_2_2(h_conv2)
#原始图像尺寸为28*28,第一轮图像缩小为14*14共32张，第二轮继续缩小为7*7共64张

#第三层全连接操作
#二维张量，第一个参数7*7*64的patch，第二个参数代表卷积有1024个
W_fc1 = weight_variable([7*7*64, 1024])
#1024个偏执量
b_fc1 = bias_veriable([1024])
#将第二层的池化结果reshape成一行，有7*7*64个数据[n, 7, 7, 64]->[n, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
#卷积操作,结果是1*1*1024，这里只有一行，所以直接采用矩阵相乘，而不是遍历相乘,第一个参数是行向量，第二个是列向量
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#drop操作，为了减少过度拟合，降低上一层某些输入的权值，防止测评曲线出现震荡
#使用占位符，由dropout自动确定scale
keep_prob = tf.placeholder(tf.float32)
#对卷积结果进行drop操作
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#第四层输出操作
#二维张量，1*1024矩阵卷积，对应我们定义的10个输出长度ys
W_fc2 = weight_variable([1024, 10])
tf.histogram_summary('outputlayer/W', W_fc2)
b_fc2 = bias_veriable([10])
tf.histogram_summary('outputlayer/b', b_fc2)

#结果层用逻辑回归softmax或者sigmoid
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#定义代价函数loss，选定优化方法优化loss
"""
交叉熵可在神经网络(机器学习)中作为损失函数，p表示真实标记的分布，q则为训练后的模型的预测标记分布，交叉熵损失函数可以衡量p与q的相似性。
交叉熵作为损失函数还有一个好处是使用sigmoid函数在梯度下降时能避免均方误差损失函数学习速率降低的问题，因为学习速率可以被输出的误差所控制。
"""

#这里使用交叉熵为损失函数
cross_entroy = -tf.reduce_sum(ys*tf.log(tf.clip_by_value(y_conv,1e-10,1.0))) #-tf.reduce_sum(ys * tf.log(y_conv))

#使用tf写好的梯度下降作为优化函数，0.5表示迭代速率，minimize表示希望交叉熵最小
train_step = tf.train.AdagradOptimizer(0.001).minimize(cross_entroy)

#开始数据训练及评测
#这句表示训练结果一组求最大值之后和实际结果是否相等
correct_perdiction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(ys, 1))
#这句表示将上面的结果向量转换成float型，然后求平均而已
arrcuray = tf.reduce_mean(tf.cast(correct_perdiction, tf.float32))

init_op = tf.initialize_all_variables()
sess.run(init_op)

#可视化
tf.scalar_summary("cross_entroy", cross_entroy)
tf.scalar_summary("arrcuray", arrcuray)

merged_summary_op = tf.merge_all_summaries()
summary_writer = tf.train.SummaryWriter('CNN.logs', sess.graph_def)

#迭代20000次
for i in range(2000):
    #每次取长度为50的一个数据片段
    batch = mnist.train.next_batch(50)

    # 执行训练
    # train_step.run(feed_dict={xs: batch[0], ys: batch[1], keep_prob: 0.5})
    sess.run(train_step, feed_dict={xs: batch[0], ys: batch[1], keep_prob: 0.5})

    #每迭代100次输出一次结果
    if i%100 == 0:
        #打印参数，填充之前的定义的占位符
        train_accuray = arrcuray.eval(feed_dict={xs:batch[0], ys:batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuray))
        summaryStr = sess.run(merged_summary_op, feed_dict={xs:batch[0], ys:batch[1], keep_prob: 1.0})
        summary_writer.add_summary(summaryStr, i)

#打印测试结果
print("test accuracy %g" % arrcuray.eval(feed_dict={xs: mnist.test.images, ys: mnist.test.labels, keep_prob: 1.0}))




