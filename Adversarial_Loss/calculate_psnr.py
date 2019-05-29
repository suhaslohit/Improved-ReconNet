import numpy as np 
import cv2
import os
import math
import scipy.io

mr = '0_25' # Choose from 0_25, 0_10, 0_04 and 0_01

inputDir = 'test_images/'
outputDir = 'recon_images/mr_' + mr + '/'

imList = os.listdir(inputDir)
psnr = np.array([])

print 'MR = ' + mr

for imName in imList:
	input_im = cv2.imread(inputDir + imName, 0)
	input_im = cv2.normalize(input_im.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
	output_im = scipy.io.loadmat(outputDir+imName+'.mat')['result']

	rmse = np.sqrt(np.mean(np.square(output_im - input_im)))
	psnr = np.append(psnr, 20*np.log10(1./rmse))

	print imName + ' : ' + str(20*np.log10(1./rmse)) + ' dB'


print '--------------'
print 'Mean PSNR: ' + str(psnr.mean()) + ' dB'