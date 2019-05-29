import tensorflow as tf
import numpy as np

import cPickle
import ipdb

# Change the name of the phi file depending on the MR

phi = np.load('phi/phi_0_25_1089.npy')
print phi.shape
phi = phi.astype(np.float32)

def new_conv_layer(bottom, filter_shape, activation=tf.identity, padding='SAME', stride=1, name=None ):
    with tf.variable_scope( name ):
        w = tf.get_variable(
                "W",
                shape=filter_shape,
                initializer=tf.random_normal_initializer(0., 0.01))
        b = tf.get_variable(
                "b",
                shape=filter_shape[-1],
                initializer=tf.constant_initializer(0.))

        conv = tf.nn.conv2d( bottom, w, [1,stride,stride,1], padding=padding)
        bias = activation(tf.nn.bias_add(conv, b))

    return bias #relu

def new_fc_layer_phi_init(bottom, output_size, name ):
    batch_size = 128
    shape = bottom.get_shape().as_list()
    print shape
    dim = np.prod( shape[1:] )
    x = tf.reshape( bottom, [-1, dim])
    input_size = dim

    with tf.variable_scope(name):
        w = tf.Variable(phi, name="W")
        b = tf.get_variable(
                "b",
                shape=[output_size],
                initializer=tf.constant_initializer(0.))
        fc = tf.nn.bias_add( tf.matmul(x, w), b)

    return fc

def new_fc_layer(bottom, output_size, name ):
    batch_size = 128
    shape = bottom.get_shape().as_list()
    dim = np.prod( shape[1:] )
    x = tf.reshape( bottom, [-1, dim])
    input_size = dim

    with tf.variable_scope(name):
        w = tf.get_variable(
                "W",
                shape=[input_size, output_size],
                initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(
                "b",
                shape=[output_size],
                initializer=tf.constant_initializer(0.))
        fc = tf.nn.bias_add( tf.matmul(x, w), b)

    return fc


def leaky_relu( bottom, leak=0.0,name=None,is_training=None):
    return tf.maximum(leak*bottom, bottom)

def build_reconstruction(images, is_train ):
    batch_size = images.get_shape().as_list()[0]

    with tf.variable_scope('GEN'):
        fc1 = new_fc_layer_phi_init(images, 1089, name="fc1")
        fc1 = tf.reshape(fc1, [batch_size, 33, 33, 1])
        conv1 = new_conv_layer(fc1, [11,11,1,64], stride=1, name="conv1" )
        bn1 = leaky_relu(conv1, is_training=is_train)
        conv2 = new_conv_layer(bn1, [1,1,64,32], stride=1, name="conv2" )
        bn2 = leaky_relu(conv2, is_training=is_train)
        conv3 = new_conv_layer(bn2, [7,7,32,1], stride=1, name="conv3")

    return bn1, bn2, conv3

def build_adversarial(images, is_train, reuse=None, keep_prob=1.0):
    with tf.variable_scope('DIS', reuse=reuse):
        conv1 = new_conv_layer(images, [4,4,1,4], stride=2, name="conv1" )
        bn1 = leaky_relu(batch_norm_wrapper(conv1, is_training=is_train))
        conv2 = new_conv_layer(bn1, [4,4,4,4], stride=2, name="conv2")
        bn2 = leaky_relu(batch_norm_wrapper(conv2, is_training=is_train))
        conv3 = new_conv_layer(bn2, [4,4,4,4], stride=2, name="conv3")
        bn3 = leaky_relu(batch_norm_wrapper(conv3, is_training=is_train))

        output = new_fc_layer( bn3, output_size=1, name='output')
        output = tf.nn.dropout(output, keep_prob)
        output_sig = tf.sigmoid(output)

    return output[:,0], output_sig[:,0]