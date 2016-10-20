# # Self Driving Car Challenge Two - CNN solution
# # based on nvidia paper

from __future__ import division
from __future__ import print_function
import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import sys, os, inspect, time
import gzip


BATCH_SIZE = 10

# Include ../scripts subfolder in path. Used to get dataset of images and angles.
scripts_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],
                    "../scripts")))
if scripts_subfolder not in sys.path:
    sys.path.insert(0, scripts_subfolder)


# Import script with dataset routines, and start getting images and angles for training, testing and validation. Edit input files in get_imgs.py file.

import get_imgs

data = get_imgs.dataset()


# Placeholders por input data [images and angles], images size is (120x160x3),
# angles size is 1
x = tf.placeholder(tf.float32, shape=[None, (120*160*3)])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

# Reshape x to a 4d tensor
# 
x_image = tf.reshape(x, [-1, 120, 160, 3])


# First Convolutional layer:
#
W_conv1 = tf.Variable(tf.truncated_normal([5,5,3,24], mean=0.0, stddev=0.1), dtype=tf.float32, name='W_conv1')
b_conv1 = tf.Variable(tf.constant(0.1, shape=[24]), dtype=tf.float32, name='bias_conv1')
h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 2, 2, 1], padding='VALID') + b_conv1)


# Second Convolution Layer
# 
W_conv2 = tf.Variable(tf.truncated_normal([5,5,24,32], mean=0.0, stddev=0.1), dtype=tf.float32, name='Weights_conv2')
b_conv2 = tf.Variable(tf.constant(0.1, shape=[32]), dtype=tf.float32, name='bias_conv2')
h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 2, 2, 1], padding='VALID') + b_conv2)

# Third Convolution Layer
# 
W_conv3 = tf.Variable(tf.truncated_normal([5,5,32,32], mean=0.0, stddev=0.1), dtype=tf.float32, name='Weights_conv3')
b_conv3 = tf.Variable(tf.constant(0.1, shape=[32]), dtype=tf.float32, name='bias_conv3')
h_conv3 = tf.nn.relu(tf.nn.conv2d(h_conv2, W_conv3, strides=[1, 2, 2, 1], padding='VALID') + b_conv3)

# Fourth Convolution Layer
# 
W_conv4 = tf.Variable(tf.truncated_normal([3,3,32,48], mean=0.0, stddev=0.1), dtype=tf.float32, name='Weights_conv4')
b_conv4 = tf.Variable(tf.constant(0.1, shape=[48]), dtype=tf.float32, name='bias_conv4')
h_conv4 = tf.nn.relu(tf.nn.conv2d(h_conv3, W_conv4, strides=[1, 2, 2, 1], padding='VALID') + b_conv4)

# Fifth Convolution Layer
# 
W_conv5 = tf.Variable(tf.truncated_normal([3,3,48,64], mean=0.0, stddev=0.1), dtype=tf.float32, name='Weights_conv5')
b_conv5 = tf.Variable(tf.constant(0.1, shape=[64]), dtype=tf.float32, name='bias_conv5')
h_conv5 = tf.nn.relu(tf.nn.conv2d(h_conv4, W_conv5, strides=[1, 1, 1, 1], padding='VALID') + b_conv5)

h_conv5_flat = tf.reshape(h_conv5, [-1, 1152])

# First Fully Connected Layer
# 
W_fcl = tf.Variable(tf.truncated_normal([1152, 512], mean=0.0, stddev=0.1), dtype=tf.float32, name='Weights_fcl')
b_fcl = tf.Variable(tf.constant(0.1, shape=[512]), dtype=tf.float32, name='bias_fcl')
h_fcl = tf.nn.relu(tf.matmul(h_conv5_flat, W_fcl) + b_fcl)

# Dropout
# 
keep_prob = tf.placeholder(tf.float32)
h_fcl_drop = tf.nn.dropout(h_fcl, keep_prob)

# Second Fully Connected Layer
# 
W_fc2 = tf.Variable(tf.truncated_normal([512, 128], mean=0.0, stddev=0.1), dtype=tf.float32, name='Weights_fc2')
b_fc2 = tf.Variable(tf.constant(0.1, shape=[128]), dtype=tf.float32, name='bias_fc2')
h_fc2 = tf.nn.relu(tf.matmul(h_fcl_drop, W_fc2) + b_fc2)

# Dropout
# 
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

# Third Fully Connected Layer
# 
W_fc3 = tf.Variable(tf.truncated_normal([128, 32], mean=0.0, stddev=0.1), dtype=tf.float32, name='Weights_fc3')
b_fc3 = tf.Variable(tf.constant(0.1, shape=[32]), dtype=tf.float32, name='bias_fc3')
h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

# Dropout
# 
h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)


# Readout Layer
# 
W_fc4 = tf.Variable(tf.truncated_normal([32, 1], mean=0.0, stddev=0.1), dtype=tf.float32, name='Weights_fc4')
b_fc4 = tf.Variable(tf.constant(0.1, shape=[1]), dtype=tf.float32, name='bias_fc4')

y_conv = tf.tanh(tf.matmul(h_fc3_drop, W_fc4) + b_fc4)


# ## Train and Evaluate model
#
mse = tf.reduce_mean(tf.squared_difference(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(mse)
correct_prediction = y_conv

accuracy = tf.cast(mse, tf.float32)

tf.scalar_summary('loss', mse)

merged = tf.merge_all_summaries()

sess = tf.Session()

train_writer = tf.train.SummaryWriter('./conv/train',sess.graph)


sess.run(tf.initialize_all_variables())

saver = tf.train.Saver()


# Check if network has previous parameters saved

# In[14]:

try:
    saver.restore(sess, './conv/save_net.ckpt')
    print("Session checkpoint restored successfully!")
except:
    print("No previous session checkpoint found.")
    pass


# ## Train

start_t = time.time()

for i in range(1348):
    images, angles = data.next_batch(BATCH_SIZE, 'train')
    images = images / 255.0;
    angles = angles.reshape(BATCH_SIZE, 1)
    run_metadata = tf.RunMetadata()
    summary, _ = sess.run([merged, train_step], feed_dict={x: images, y_: angles, keep_prob: 0.5}, run_metadata=run_metadata)
    train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
    train_writer.add_summary(summary, i)
    if (i%10) == 0:
        lapse_t = time.time() - start_t
        saver.save(sess, './conv/save_net.ckpt')
        images, angles = data.next_batch(BATCH_SIZE, 'test')
        images = images / 255.0;
        angles = angles.reshape(BATCH_SIZE,1)
        train_accuracy = sess.run(accuracy, feed_dict={x:images, y_: angles, keep_prob: 1.0})
        print("Step: %03d lapse time: %2.4f scs.\ttest error: %g"%(i,lapse_t, train_accuracy))
        start_t = time.time()

print("Finish !!!!")
saver.save(sess, './conv/save_net.ckpt')
images, angles = data.next_batch(1273, 'validation')
images = images / 255.0
angles = angles.reshape(1273,1)
error, predict_angles = sess.run([accuracy,y_conv], feed_dict={x: images, y_: angles, keep_prob: 1.0})
print("Validation error: %g"%(error))

sess.close()






