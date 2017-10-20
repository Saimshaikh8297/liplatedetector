from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import numpy as np
import os
import cv2
import tensorflow as tf
from scipy import misc
import glob
from skimage import io,transform,color
#import license_detect as ld


train_dataset='eng/Fnt'
X, Y = tflearn.data_utils.image_preloader (train_dataset, image_shape=(28, 28), mode='folder', normalize=True, grayscale=True, categorical_labels=True, files_extension=None, filter_channel=False)
# #
# # #test_dataset='eng/Fnt'
# # #X_test, Y_test = tflearn.data_utils.image_preloader (test_dataset, image_shape=(28, 28), mode='folder', normalize=True, grayscale=True, categorical_labels=True, files_extension=None, filter_channel=False)
# #
# # #print(X.array)
X = np.reshape(X,(-1,28,28,1))
# X_test = np.reshape(X_test, (-1, 28, 28, 1))
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# # Building convolutional network
network = input_data(shape=[None, 28, 28, 1],data_preprocessing=img_prep, name='input')
network = conv_2d(network, 32, 2, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = conv_2d(network, 64, 2, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = fully_connected(network, 1024, activation='softmax')
network = dropout(network, 0.4)
network = fully_connected(network, 36, activation='softmax')
# # network = dropout(network, 0.8)
# #network = fully_connected(network, 36, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.01,
                     loss='categorical_crossentropy', name='target')

model = tflearn.DNN(network, tensorboard_verbose=3)
# model.load('model_saved.tflearn')
# # # Training
#
# model.fit({'input': X}, {'target': Y}, n_epoch=10,
#            validation_set=0.3,
#              snapshot_step=500, show_metric=True, run_id='acb')

model.load('model_saved1.tflearn')

folder='testfolder'

def load_img_from_folder(folder):
    images=[]
    for file in os.listdir(folder):
        c=io.imread(os.path.join(folder,file), as_grey=True ,flatten=False)
        c=color.rgb2gray(c)
        #c=io.imread('test2.png', as_grey=True ,flatten=False)
        #c=transform.resize(c,(128,128))
        c=transform.resize(c,(28,28))
        c=np.array([c], np.float32)
        c=np.reshape(c,(28,28,-1))
        pre = model.predict([c])
        #print(file)
        a=np.where(pre==pre.max())
        #print(a[1][0])
        images.append(a[1][0])

    return images


val = load_img_from_folder(folder)
