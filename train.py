import tensorflow as tf
import tensorlayer as tl
import numpy as np
import os

flags = tf.app.flags
flags.DEFINE_string("model", "dnn", "选择训练模型")
flags.DEFINE_integer("epoch", 1000, "迭代次数")
flags.DEFINE_float("learning_rate", 0.001, "学习速率")
flags.DEFINE_integer("print_freq", 5, "每多少次迭代打印一次")
flags.DEFINE_integer("batch_size", 100, "每次迭代训练样本个数")
FLAGS = flags.FLAGS


sess = tf.InteractiveSession()

# 加载测试样本
X_Val = np.loadtxt(os.path.join("data","1 (2).txt"), delimiter=" ")
y_Val = np.loadtxt(os.path.join("data","2 (2).txt"), delimiter=" ")
X_Val = np.asarray(X_Val, dtype=np.float32)
y_Val = np.asarray(y_Val, dtype=np.int64)

# 选择模型
if FLAGS.model == "cnn":
    X_Val = np.array(X_Val).reshape(X_Val.shape[0], 8, 8, 1)
    x = tf.placeholder(tf.float32, shape=[None, 8, 8, 1], name="x")
    y_ = tf.placeholder(tf.int64, shape=[None, ], name="y_")

    network = tl.layers.InputLayer(x, name="input_layer")
    network = tl.layers.Conv2dLayer(network, act=tf.nn.relu, shape=[3, 3, 1, 16],
                                    strides=[1, 1, 1, 1], padding="SAME", name="cnn_layer1")  # output:(?,6,6,16)
    network = tl.layers.PoolLayer(network, ksize=[1, 2, 2, 1], strides=[
                                  1, 2, 2, 1], padding="SAME", pool=tf.nn.max_pool, name="pool_layer1")  # output:(?,3,3,16)
    network = tl.layers.FlattenLayer(
        network, name="faltten_layer")  # output:(?,128)
    network = tl.layers.DropoutLayer(network, keep=0.5, name="drop1")
    network = tl.layers.DenseLayer(
        network, n_units=256, act=tf.nn.relu, name="relu")
    network = tl.layers.DropoutLayer(network, keep=0.5, name="drop2")
    network = tl.layers.DenseLayer(
        network, n_units=64, act=tf.identity, name="output_layer")
elif FLAGS.model == "dnn":
    X_Val = np.array(X_Val).reshape(X_Val.shape[0], 64)
    x = tf.placeholder(tf.float32, shape=[None, 64], name="x")
    y_ = tf.placeholder(tf.int64, shape=[None, ], name="y_")

    network = tl.layers.InputLayer(x, name="input_layer")
    network = tl.layers.DropoutLayer(network, keep=0.8, name="drop1")
    network = tl.layers.DenseLayer(
        network, n_units=800, act=tf.nn.relu, name="relu1")
    network = tl.layers.DropoutLayer(network, keep=0.7, name="drop2")
    network = tl.layers.DenseLayer(
        network, n_units=900, act=tf.nn.relu, name="relu2")
    network = tl.layers.DropoutLayer(network, keep=0.7, name="drop3")
    network = tl.layers.DenseLayer(
        network, n_units=900, act=tf.nn.relu, name="relu3")
    network = tl.layers.DropoutLayer(network, keep=0.5, name="drop4")
    network = tl.layers.DenseLayer(
        network, n_units=64, act=tf.identity, name="output_layer")
else:
    print("error input!")

y = network.outputs
cost = tl.cost.cross_entropy(y, y_)
correct_prediction = tf.equal(tf.argmax(y, 1), y_)
y_op = tf.argmax(tf.nn.softmax(y), 1)
# 定义 optimizer
train_params = network.all_params
train_op = tf.train.AdamOptimizer(
    learning_rate=FLAGS.learning_rate).minimize(cost, var_list=train_params)

y_op = tf.argmax(tf.nn.softmax(y), 1)

# 初始化 session 中的所有参数
tl.layers.initialize_global_variables(sess)


if __name__ == "__main__":
    # 训练模型
    for i in range(5):
        try:
            params = tl.files.load_npz(path="models",name="\\model.npz")
            tl.files.assign_params(sess, params, network)
            print("成功读取模型！")
        except:
            print("读取模型失败!")

        X_train = np.loadtxt(os.path.join("data", str(
            1000 * i + 1000)+"_1.txt"), delimiter=" ")
        y_train = np.loadtxt(os.path.join("data", str(
            1000 * i + 1000)+"_2.txt"), delimiter=" ")
        if FLAGS.model == "cnn":
            X_train = np.array(X_train).reshape(X_train.shape[0], 8, 8, 1)
        else:
            X_train = np.array(X_train).reshape(X_train.shape[0], 64)
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.int64)
        tl.utils.fit(sess, network, train_op, cost, X_train, y_train, x, y_,
                     n_epoch=FLAGS.epoch, print_freq=FLAGS.print_freq, batch_size=FLAGS.batch_size, X_val=X_Val, y_val=y_Val)
        tl.files.save_npz(network.all_params, path = "models",name="\\model.npz")
