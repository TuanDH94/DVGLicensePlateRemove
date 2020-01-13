import tensorflow as tf
import sys


def drop_out(x, keep_prob):
    return tf.nn.dropout(x, keep_prob=keep_prob)


def weight(name, shape):
    return tf.get_variable(name=name,
                           initializer=tf.truncated_normal(shape, stddev=0.1),
                           dtype=tf.float32)


def bias(name, shape):
    return tf.get_variable(name=name,
                           initializer=tf.constant(0.1, shape=shape))


def conv2d(name, x, shape):
    conv_w = weight(name=name+"_w", shape=shape)
    conv_b = bias(name=name+"_b", shape=[shape[-1]])
    conv = tf.nn.conv2d(x, conv_w, [1, 1, 1, 1], padding='SAME')
    conv = tf.nn.bias_add(conv, conv_b)
    conv = tf.nn.relu(conv)
    return conv


def max_pool2d(name, x, kernel, strides):
    return tf.nn.max_pool(value=x,
                          ksize=[1, kernel[0], kernel[1], 1],
                          strides=[1, strides[0], strides[1], 1],
                          padding='SAME',
                          name=name+"_pool")


def fcn(name, x, shape, activation=True):
    fcn_w = weight(name=name+"_w", shape=shape)
    fcn_b = bias(name=name+"_b", shape=[shape[-1]])
    fcn = tf.matmul(x, fcn_w)
    fcn = tf.nn.bias_add(fcn, fcn_b)
    if activation:
        return tf.nn.relu(fcn)
    else:
        return fcn


def simple_model(x, height, width, channels, num_classes):

    float_x = x / 255

    with tf.name_scope('reshape'):
        y = tf.reshape(float_x, [-1, height, width, channels])

    with tf.name_scope('conv1'):
        y = conv2d('conv1', y, [5, 5, channels, 32])
        y = max_pool2d('conv1', y , [2, 2], [2, 2])

    with tf.name_scope('conv2'):
        y = conv2d('conv2', y, [5, 5, 32, 64])
        y = max_pool2d('conv2', y, [2, 2], [2, 2])

    flat_size = (height//4) * (width//4) * 64
    y = tf.reshape(y, [-1, flat_size])

    with tf.name_scope('fcn1'):
        y = fcn('fcn1', y, [flat_size, 1024])

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        y = drop_out(y, keep_prob)

    with tf.name_scope('fcn2'):
        y = fcn('fcn2', y, [1024, num_classes], activation=False)

    return y, keep_prob


class CNNModel:

    def __init__(self, height, width, channels, num_classes):
        self.sess = tf.Session()
        self.height = height
        self.width = width
        self.channels = channels
        self.num_classes = num_classes
        self.x = tf.placeholder(tf.float32, [None, self.height * self.width * self.channels])
        self.y_ = tf.placeholder(tf.float32, [None, self.num_classes])
        self.y, self.keep_prob = simple_model(self.x , self.height, self.width, self.channels, self.num_classes)
        with tf.name_scope('loss'):
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y)
            self.cross_entropy = tf.reduce_mean(self.cross_entropy)
        with tf.name_scope('adam_optimizer'):
            self.train_op = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
        with tf.name_scope('accuracy'):
            self.accuracy = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
            self.accuracy = tf.cast(self.accuracy, tf.float32)
            self.accuracy = tf.reduce_mean(self.accuracy)
        self.sess.run(tf.global_variables_initializer())

    def save_model(self, save_path, global_step):
        sys.stdout.write('Saving model to {}...'.format(save_path))
        saver = tf.train.Saver()
        saver.save(sess=self.sess, save_path=save_path, global_step=global_step)
        print('Finished!')

    def load_model(self, load_path, tracking):
        if tracking:
            sys.stdout.write('Loading model from {}...'.format(load_path))
        loader = tf.train.Saver()
        loader.restore(sess=self.sess, save_path=load_path)
        if tracking:
            print('Finished!')

    def sgd(self, x, y_):
        self.train_op.run(session=self.sess, feed_dict={self.x: x, self.y_: y_, self.keep_prob: 0.5})

    def eval(self, x, y_):
        return self.accuracy.eval(session=self.sess, feed_dict={self.x: x, self.y_: y_, self.keep_prob: 1.0})

    def predict(self, x):
        output = self.y.eval(session=self.sess, feed_dict={self.x: x, self.keep_prob: 1.0})
        output = tf.nn.softmax(output)
        return self.sess.run(output)


import Config
from utils_segment.DataReader import DirectoryScanner, GRAY, BINARY

if __name__ == '__main__':
    neg_dir = r'E:\DVG.Data\dvg\PlateLocalization\PlateCharacters\0'
    pos_dir = r'E:\DVG.Data\dvg\PlateLocalization\PlateCharacters\1_refined'
    epoches = 5000
    batch_size = 50
    scanner = DirectoryScanner(scanner_type=BINARY)
    scanner.load_train_data(pos_dir, neg_dir, limit=30000)
    model = CNNModel(Config.char_height, Config.char_width, 1, 2)

    train_accuracy = 0

    for i in range(epoches):
        batch = scanner.next_train_batch(batch_size)
        train_accuracy += model.eval(batch[0], batch[1])
        if (i+1) % 50 == 0:
            train_accuracy /= 50
            print('Step {}, train accuracy: {}'.format(i+1, train_accuracy))
            train_accuracy = 0
        if (i+1) % 1000 == 0:
            model.save_model(save_path='./binary_models/plate_character_detect_trained.ckpt', global_step=i+1)
        model.sgd(batch[0], batch[1])


