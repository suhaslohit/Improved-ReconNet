import numpy as np 
import tensorflow as tf 
import cv2
import os
import sys
import csgan
import scipy.io

# Change these values before running code. 
# MR = 0.01, 0.04, 0.10, 0.25 and corresponding m = 10, 43, 109, 272

checkpointPath = 'checkpoints_final/mr_0_25_79000'
inputDir = 'test_images/'
matdir = 'recon_images/mr_0_25/'
phi = np.load('phi/phi_0_25_1089.npy')
blockSize = 33
m = 272
batch_size = 1

imList = os.listdir(inputDir)
print imList

with tf.Graph().as_default():

	images_tf = tf.placeholder( tf.float32, [batch_size, 33, 33, 1], name="images")
	cs_meas = tf.placeholder( tf.float32, [batch_size, 1, m, 1], name='cs_meas')
	is_train = tf.placeholder( tf.bool )

	bn1, bn2, reconstruction_ori = csgan.build_reconstruction(cs_meas, is_train)

	summary = tf.merge_all_summaries()
	saver = tf.train.Saver()
	sess = tf.Session()

	saver.restore(sess, checkpointPath)

	psnr = np.array([])


	for imName in imList:
		# Read image
		im = cv2.imread(inputDir + imName,0)
		im = cv2.normalize(im.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
		[height, width] = im.shape

		# Determine the size of zero pad
		rowPad = blockSize - (height % blockSize)
		colPad = blockSize - (width % blockSize)

		# Do zero padding
		imPad = np.concatenate((im, np.zeros([rowPad, width])), axis=0)
		imPad = np.concatenate((imPad, np.zeros([height+rowPad, colPad])),axis=1)
		print imPad.shape

		numBlocksRow = (height + rowPad)/blockSize
		numBlocksCol = (width + colPad)/blockSize

		outputImPad = np.zeros([height+rowPad, width+colPad])

		for i in xrange(numBlocksRow):
			for j in xrange(numBlocksCol):

				# Break into blocks
				block = imPad[i*blockSize:(i+1)*blockSize,j*blockSize:(j+1)*blockSize]
				block = np.hstack(block)
				block = np.reshape(block, [blockSize*blockSize, 1])
				blockCS = phi.dot(block)
				blockCS = np.reshape(blockCS, [1, 1, m, 1]) # Reshape to 4D tensor
				blockIm = np.reshape(block, [1,blockSize,blockSize,1])

				# Feed blocks to the trained network
				reconstruction_ori_val = sess.run([reconstruction_ori], feed_dict={cs_meas: blockCS, is_train: False})
				reconstruction_op = np.reshape(reconstruction_ori_val, [33,33])

				# Re-arrange output into image
				outputImPad[i*blockSize:(i+1)*blockSize,j*blockSize:(j+1)*blockSize] = reconstruction_op
				outputIm = outputImPad[0:height,0:width]

		rmse = np.sqrt(np.mean(np.square(outputIm - im)))
		psnr = np.append(psnr, 20*np.log10(1./rmse))
		a = {}
		a['result'] = outputIm
		scipy.io.savemat(matdir+imName, a)

	print psnr
	print '--------------'
	print psnr.mean()