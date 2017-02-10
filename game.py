import sys
import tensorflow as tf
import tensorlayer as tl
import numpy as np

flags = tf.app.flags
flags.DEFINE_string("model", "cnn", "选择训练模型")
FLAGS = flags.FLAGS

# 下棋主函数
# 输入：一个64位向量
# 输出：一个64位向量，以模型认为最有可能的顺序排列


def xiaqi(map):
    if FLAGS.model == "cnn":
        map = np.array(map).reshape(1, 8, 8, 1)
    else:
        map = np.array(map).reshape(1,64)    
    out = tl.utils.predict(sess, network, map, x, y_op)
    out = np.array(np.argsort(-out)).reshape(64)
    return out

# 加载模型数据
sess = tf.InteractiveSession()
if FLAGS.model == "cnn":
    x = tf.placeholder(tf.float32, shape=[None, 8, 8, 1], name="x")
    network = tl.layers.InputLayer(x, name="input_layer")
    network = tl.layers.Conv2dLayer(network, act=tf.nn.relu, shape=[3, 3, 1, 16], strides=[
        1, 1, 1, 1], padding="SAME", name="cnn_layer1")  # output:(?,6,6,16)
    network = tl.layers.PoolLayer(network, ksize=[1, 2, 2, 1], strides=[
        1, 2, 2, 1], padding="SAME", pool=tf.nn.max_pool, name="pool_layer1")  # output:(?,3,3,16)
    network = tl.layers.FlattenLayer(
        network, name="faltten_layer")  # output:(?,144)
    network = tl.layers.DropoutLayer(network, keep=0.5, name="drop1")
    network = tl.layers.DenseLayer(
        network, n_units=256, act=tf.nn.relu, name="relu")
    network = tl.layers.DropoutLayer(network, keep=0.5, name="drop2")
    network = tl.layers.DenseLayer(
        network, n_units=64, act=tf.identity, name="output_layer")
elif FLAGS.model == "dnn":
    x = tf.placeholder(tf.float32, shape=[None, 64], name="x")
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

sess.run(tf.global_variables_initializer())
params = tl.files.load_npz(path = "models",name="\\model.npz")
tl.files.assign_params(sess, params, network)
y = network.outputs
y_op = tf.nn.softmax(y)


# 初始化棋盘数据
map = [[0 for x in range(8)] for y in range(8)]
map[3][3] = map[4][4] = 1
map[3][4] = map[4][3] = 2
# 初始化移动数据
mv = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]


# 更新棋盘数据
# 输入：坐标、颜色
def work(row, col, t_color):
    map[row][col] = t_color
    for k in range(8):
        t_row = row + mv[k][0]
        t_col = col + mv[k][1]
        while (t_row >= 0 and t_col >= 0 and t_row < 8 and t_col < 8 and map[t_row][t_col] == 3 - t_color):
            t_row += mv[k][0]
            t_col += mv[k][1]
        if (t_row >= 0 and t_col >= 0 and t_row < 8 and t_col < 8 and map[t_row][t_col] == t_color):
            t_row = row + mv[k][0]
            t_col = col + mv[k][1]
            while (t_row >= 0 and t_col >= 0 and t_row < 8 and t_col < 8 and map[t_row][t_col] == 3 - t_color):
                map[t_row][t_col] = t_color
                t_row += mv[k][0]
                t_col += mv[k][1]

# 检测位置合法性
# 输入：坐标、颜色
# 输出：一个bool值代表是否合法


def check(row, col, t_color):
    if map[row][col] != 0:
        return False
    for k in range(8):
        t_row = row + mv[k][0]
        t_col = col + mv[k][1]
        while (t_row >= 0 and t_col >= 0 and t_row < 8 and t_col < 8 and map[t_row][t_col] == 3 - t_color):
            t_row += mv[k][0]
            t_col += mv[k][1]
        if (t_row >= 0 and t_col >= 0 and t_row < 8 and t_col < 8 and
                map[t_row][t_col] == t_color and map[t_row - mv[k][0]][t_col - mv[k][1]] == 3 - t_color):
            return True
    return False


# 选择下棋位置
# 输出：一个代表位置的int，如果不存在，则返回-1
def get_place():
    map_flat = []
    for l in map:
        for x in l:
            map_flat.append(x)
    if color == 2:
        for i in range(64):
            if map_flat[i] == 2:
                map_flat[i] = 1
            if map_flat[i] == 1:
                map_flat[i] == -1
    else:
        for i in range(64):
            if map_flat[i] == 2:
                map_flat[i]=-1

    ans = xiaqi(map_flat)
    for x in ans:
        row = x // 8
        col = x - row * 8
        if check(row, col, color):
            return x
    return -1

q = input()
sys.stdin.flush()

if q == 'WHITE':
    color = 1
    print("NO")
    sys.stdout.flush()
else:
    color = 2
    step = get_place()
    row = step // 8
    col = step - row * 8
    print(chr(row + ord('1')) + chr(col + ord('A')))
    sys.stdout.flush()
    work(row, col, color)

while True:
    s = input()
    sys.stdin.flush()
    if s != "NO":
        row = int(ord(s[0]) - ord('1'))
        col = int(ord(s[1]) - ord('A'))
        work(row, col, 3 - color)
    step = get_place()
    if step == -1:
        print("NO")
        sys.stdout.flush()
        continue
    row = step // 8
    col = step - row * 8
    print(chr(row + ord('1')) + chr(col + ord('A')))
    sys.stdout.flush()
    work(row, col, color)
