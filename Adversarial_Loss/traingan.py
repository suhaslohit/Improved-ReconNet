import ipdb
import os
from glob import glob
import pandas as pd
import numpy as np
import cv2
import csgan
import h5py
import tensorflow as tf

n_epochs = 200
weight_decay_rate =  0.00001
weight_decay_rate_dis =  0.001
momentum = 0.9
batch_size = 128
lambda_recon = 1
lambda_adv = 0.001

max_iters = 100000

training_data = 'train_0_04.h5'
validation_data = 'test_0_04.h5'
checkpoint_file = 'checkpoints_final/mr_0_04/'

m = 43

with h5py.File(training_data) as hf:
    print hf.keys()
    data = np.array(hf.get('data'))
    print data.shape
    label = np.array(hf.get('label'))
    print label.shape

with h5py.File(validation_data) as hf:
    print hf.keys()
    valid_data = np.array(hf.get('data'))
    print valid_data.shape
    valid_label = np.array(hf.get('label'))
    print valid_label.shape

with tf.Graph().as_default():

    is_train = tf.placeholder( tf.bool )
    learning_rate = tf.placeholder( tf.float32, [])
    images_tf = tf.placeholder( tf.float32, [batch_size, 33, 33, 1], name="images")    
    cs_meas = tf.placeholder( tf.float32, [batch_size, 1, m, 1], name='cs_meas')
    keep_prob = tf.placeholder(tf.float32, [])

    
    labels_D = tf.concat( 0, [tf.ones([batch_size]), tf.zeros([batch_size])] )
    labels_G = tf.ones([batch_size])
    

    bn1, bn2, bn3, bn4, bn5, reconstruction_ori  = csgan.build_reconstruction(cs_meas, is_train)
    loss_recon = tf.div(tf.reduce_sum(tf.square(tf.sub(images_tf, reconstruction_ori))), 2.*batch_size) 
    
    
    adversarial_pos, adversarial_pos_sig = csgan.build_adversarial(images_tf, is_train)
    adversarial_neg, adversarial_neg_sig = csgan.build_adversarial(reconstruction_ori, is_train, reuse=True) # I changed this from reconstruction to reconstruction_ori. No idea which is right
    adversarial_all = tf.concat(0, [adversarial_pos, adversarial_neg])
    

    loss_adv_D = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(adversarial_all, labels_D))
    loss_adv_G = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(adversarial_neg, labels_G))

    loss_G = loss_recon * lambda_recon + loss_adv_G * lambda_adv
    loss_D = loss_adv_D
    

    var_G = filter( lambda x: x.name.startswith('GEN'), tf.trainable_variables())
    var_D = filter( lambda x: x.name.startswith('DIS'), tf.trainable_variables())

    for v in var_G:
        print v.name

    for v in var_D:
        print v.name

    W_G = filter(lambda x: x.name.endswith('W:0'), var_G)
    W_D = filter(lambda x: x.name.endswith('W:0'), var_D)

    loss_D += weight_decay_rate_dis * tf.reduce_mean(tf.pack( map(lambda x: tf.nn.l2_loss(x), W_D)))


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Initial training of generator
    optimizer_G_initial = tf.train.GradientDescentOptimizer(learning_rate)
    train_op_G_initial = optimizer_G_initial.minimize(loss_recon, var_list=var_G)

    

    optimizer_G = tf.train.AdamOptimizer( learning_rate=learning_rate, beta1=0.5 )
    train_op_G = optimizer_G.minimize(loss_G, var_list=var_G)


    optimizer_D = tf.train.AdamOptimizer( learning_rate=learning_rate, beta1=0.5 )
    train_op_D = optimizer_G.minimize(loss_D, var_list=var_D)
    
    
    

    saver = tf.train.Saver(max_to_keep=1000)

    sess = tf.Session(config=config)
    init = tf.initialize_all_variables()
    sess.run(init)
    

    iters = 0

    loss_D_val = 0.
    loss_G_val = 0.

    for iters in xrange(max_iters):
        batch_idx = iters % 170
        valid_batch_idx = iters % 25

        if iters <  90000:
            learning_rate_val = 0.0001
        elif iters >= 90000 and iters < 95000:
            learning_rate_val = 0.00001
        else:
            learning_rate_val = 0.000001

    	# Generative Part is updated every iteration
        _, loss_G_val, adv_pos_val, adv_neg_val, loss_recon_val, loss_adv_G_val, recon_ori_vals, bn1_val,bn2_val,bn3_val,bn4_val,bn5_val= sess.run(
                [train_op_G, loss_G, adversarial_pos, adversarial_neg, loss_recon, loss_adv_G, reconstruction_ori, bn1,bn2,bn3,bn4,bn5],
                feed_dict={
                    images_tf: label[batch_idx*batch_size: (batch_idx+1)*batch_size, :, :, :],
                    cs_meas: data[batch_idx*batch_size: (batch_idx+1)*batch_size, :, :, :],
                    learning_rate: learning_rate_val,
                    is_train: True,
                    keep_prob: 0.5
                    })

        _, loss_G_val, adv_pos_val, adv_neg_val, loss_recon_val, loss_adv_G_val, recon_ori_vals, bn1_val,bn2_val,bn3_val,bn4_val,bn5_val = sess.run(
                [train_op_G, loss_G, adversarial_pos, adversarial_neg, loss_recon, loss_adv_G, reconstruction_ori, bn1,bn2,bn3,bn4,bn5],
                feed_dict={
                    images_tf: label[batch_idx*batch_size: (batch_idx+1)*batch_size, :, :, :],
                    cs_meas: data[batch_idx*batch_size: (batch_idx+1)*batch_size, :, :, :],
                    learning_rate: learning_rate_val,
                    is_train: True,
                    keep_prob: 0.5
                    })
        
        _, loss_D_val, adv_pos_val, adv_pos_val_sig, adv_neg_val, adv_neg_val_sig = sess.run(
                [train_op_D, loss_D, adversarial_pos, adversarial_pos_sig, adversarial_neg, adversarial_neg_sig],
                feed_dict={
                    images_tf: label[batch_idx*batch_size: (batch_idx+1)*batch_size, :, :, :],
                    cs_meas: data[batch_idx*batch_size: (batch_idx+1)*batch_size, :, :, :],
                    learning_rate: learning_rate_val/100,
                    is_train: True,
                    keep_prob: 0.5
                    })
        
        # Validation:
        if iters % 10 == 0:

            loss_recon_test, adv_pos_val, adv_pos_val_sig, adv_neg_val, adv_neg_val_sig =  sess.run([loss_recon, adversarial_pos, adversarial_pos_sig, adversarial_neg, adversarial_neg_sig],
                                                                                             feed_dict={images_tf : valid_label[valid_batch_idx*batch_size: (valid_batch_idx+1)*batch_size, :, :, :], 
                                                                                                        cs_meas : valid_data[valid_batch_idx*batch_size: (valid_batch_idx+1)*batch_size, :, :, :], 
                                                                                                        is_train:False,
                                                                                                        keep_prob: 1.0
                                                                                                        })
            print "Iter:", iters, "Gen Loss:", loss_G_val, "Recon Loss:", loss_recon_test, "Gen ADV Loss:", loss_adv_G_val,  "Dis Loss:", loss_D_val
            print adv_pos_val_sig.mean(), adv_neg_val_sig.mean()
        if (iters % 100 == 0) or (iters + 1) == max_iters:
        	saver.save(sess, checkpoint_file+str(iters))
    