# *.* coding:utf-8 *.*
# @Time    : 18-6-4 下午4:24
# @Author  : Lxy
# @File    : train_2017acdc.py
# @Software: PyCharm Community Edition

import dicom, cv2, re
import os, fnmatch, sys
import numpy as np
import itertools
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from fcn_model import fcn_model
from helpers import center_crop, lr_poly_decay, get_SAX_SERIES
import matplotlib.pyplot as plt
from PIL import Image
import cv2

#   Choose GPU
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


#   model canshu
crop_size = 512
input_shape = (crop_size, crop_size, 3)
num_classes = 4
# weight_path = 'model_logs/0111_sunnybrook_i_epoch_95.h5'
weight_path = None
model = fcn_model(input_shape, num_classes, weights=weight_path)
epochs = 20
mini_batch_size = 1

npy_path = '/media/lxy/F240900A408FD42F/'
print '#'*30
print 'Loading images and labels'
images = np.load(npy_path+'images_200.npy')
masks = np.load(npy_path+'masks_200.npy')
print 'Loading is readly, ', len(images), 'images has benn prepared\n'

max_iter = (len(images) / mini_batch_size) * epochs
curr_iter = 0
base_lr = K.eval(model.optimizer.lr)
lrate = lr_poly_decay(model, base_lr, curr_iter, max_iter, power=0.5)
for e in range(epochs):
    train_generator = itertools.izip(images, masks)
    print('\nMain Epoch {:d}\n'.format(e + 1))
    print('Learning rate: {:6f}\n'.format(lrate))
    train_result = []
    for iteration in range(len(images) / mini_batch_size):
        img, mask = next(train_generator)
        res = model.train_on_batch(np.reshape(img, (1,512,512,3)), np.reshape(mask, (1,512,512,4)))
        curr_iter += 1
        lrate = lr_poly_decay(model, base_lr, curr_iter,
                              max_iter, power=0.5)
        train_result.append(res)
    train_result = np.asarray(train_result)
    train_result = np.mean(train_result, axis=0).round(decimals=10)
    print('Train result {:s}:\n{:s}'.format(model.metrics_names, train_result))
    if (e+1) % 5 == 0 or e+1 == epochs:
        save_file = '_'.join(['mul_seg_acdc',
                              'epoch', str(e + 1)]) + '.h5'
        if not os.path.exists('mul_seg_acdc_logs'):
            os.makedirs('mul_seg_acdc_logs')
        save_path = os.path.join('mul_seg_acdc_logs', save_file)
        print('\nSaving model weights to {:s}'.format(save_path))
        model.save_weights(save_path)