import tensorflow as tf
from tflearn.layers.core import dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import numpy as np
import cv2
from tflearn.data_utils import shuffle, to_categorical


# height = 402
# width = 539
max_num = 5
num_examples = 2500
batch_size = 20
train_path = '/media/violet/New Volume/dell/Documents/Files/学习/courses/CS/771/videos/people/people_train/people_train_pc.txt'
n = -1

def load_data(target_path):
    images = []
    labels = []
    with open(target_path, 'r') as f:
            for l in f.readlines():
                l = l.strip('\n').split()
                images.append(l[0].ljust(18)+ l[1])
                labels.append(int(l[2]))
    return images, labels


def next_batch(images, labels, batch_size):
    global n
    n += 1
    max_num = np.max(labels) + 1
    batch_x = []
    for index in range(batch_size):
        image = cv2.imread(images[index])
        image = cv2.resize(image, (128, 128))
        height, width, channels = image.shape
        image = np.reshape(image, (height * width * channels))
        batch_x.append(image)

    batch_y = labels[0 * batch_size : batch_size]
    batch_y = to_categorical(batch_y, max_num)

    return batch_x, batch_y


images, labels = load_data(train_path)

images, labels = shuffle(images, labels)


X = tf.placeholder(shape=(None, 128 * 128 * 3), dtype=tf.float32)
Y = tf.placeholder(shape=(None, max_num), dtype=tf.float32)

X_input = tf.reshape(X, [-1, 128, 128, 3])

network = conv_2d(X_input, 8, 9, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 8, 5, activation='relu')
network = max_pool_2d(network, 1)
network = fully_connected(network, 128, activation='relu')
network = fully_connected(network, 128, activation='relu')
network = dropout(network, 0.9)
network = fully_connected(network, max_num, activation='softmax')


# Defining other ops using Tensorflow
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(2): # 2 epochs
        avg_cost = 0.
        total_batch = int(num_examples/batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = next_batch(images, labels, batch_size)
            sess.run(optimizer, feed_dict={X: batch_xs, Y: batch_ys})
            cost = sess.run(loss, feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += cost/total_batch
            if i % 20 == 0:
                print("Epoch:", '%03d' % (epoch+1), "Step:", '%03d' % i, "Loss:", str(cost))





