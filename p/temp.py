#!encoding=utf-8
import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("Mnist_data/", one_hot=True)

x = tf.placeholder("float", [None, 784])

# 权重和偏置量
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 模型
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 定义损失函数 交叉熵
# y_输入正确值
y_ = tf.placeholder("float", [None, 10])

# 交叉熵
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# 训练
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


# 初始化变量
init = tf.initialize_all_variables()

# 启动模型
sess = tf.Session()
sess.run(init)

# 开始训练模型
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 是否正确预测
correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))

# 把布尔值换成浮点值，取平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
