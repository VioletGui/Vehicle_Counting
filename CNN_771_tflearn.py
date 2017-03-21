import tflearn
# import tensorflow as tf
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

# Data loading and preprocessing
height = 402
width = 539
max_num = 5

from tflearn.data_utils import image_preloader
X, Y = image_preloader(
    '/media/violet/New Volume/dell/Documents/Files/学习/courses/CS/771/videos/people/people_train',
    image_shape=(height, width),   mode='folder', categorical_labels=True,   normalize=True)
# X_test, Y_test = image_preloader(
#     '/media/violet/New Volume/dell/Documents/Files/学习/courses/CS/771/videos/people/people_test',
#     image_shape=(height, width),   mode='folder', categorical_labels=True,   normalize=True)


# Convolutional network building
network = input_data(shape=[None, width, height, 3])
# X_input = tf.placeholder(shape=(None, width * height), dtype=tf.float32)
# Y_input = tf.placeholder(shape=(None, max_num), dtype=tf.float32)
#
# network = tf.reshape(X, [-1, width, height, 1])
network = conv_2d(network, 8, 9, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 8, 5, activation='relu')
network = max_pool_2d(network, 1)
network = fully_connected(network, 128, activation='relu')
network = fully_connected(network, 128, activation='relu')
network = dropout(network, 0.9)
network = fully_connected(network, max_num, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Train using classifier
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit(X, Y, n_epoch=2, shuffle=True, validation_set=(X, Y),
          show_metric=True, batch_size=30, run_id='counting_people')
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(network, Y_input))
# optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
#
# # Initializing the variables
# init = tf.initialize_all_variables()
#
# # Launch the graph
# with tf.Session() as sess:
#     sess.run(init)
#
#     batch_size = 20
#     for epoch in range(2): # 2 epochs
#         avg_cost = 0.
#         total_batch = int(num_examples/batch_size)
#         for i in range(total_batch):
#             batch_xs, batch_ys = image_preloader(
#                 '/media/violet/New Volume/dell/Documents/Files/学习/courses/CS/771/videos/people/people_train',
#                 image_shape=(height, width), mode='folder', categorical_labels=True, normalize=True)
#             sess.run(optimizer, feed_dict={X_input: batch_xs, Y_input: batch_ys})
#             cost = sess.run(loss, feed_dict={X_input: batch_xs, Y_input: batch_ys})
#             avg_cost += cost/total_batch
#             if i % 20 == 0:
#                 print("Epoch:", '%03d' % (epoch+1), "Step:", '%03d' % i,
#                         "Loss:", str(cost))




