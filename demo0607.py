# *.* coding:utf-8 *.*
# @Time    : 18-6-7 下午2:36
# @Author  : Lxy
# @File    : demo0607.py
# @Software: PyCharm Community Edition

import cv2
import numpy as np
import itertools

npy_path = '/media/lxy/F240900A408FD42F/'
print '#'*30
print 'Loading images and labels'
images = np.load(npy_path+'images_200.npy')
masks = np.load(npy_path+'masks_200.npy')
print 'Loading is readly, ', len(images), 'images has benn prepared\n'
k = 0

for j in range(5):
    train_generator = itertools.izip(images, masks)
    for i in range(200):
        k += 1
        image, mask = next(train_generator)
        print(image.shape)
print k