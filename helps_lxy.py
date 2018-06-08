# *.* coding:utf-8 *.*
# @Time    : 18-6-7 下午3:18
# @Author  : Lxy
# @File    : helps_lxy.py
# @Software: PyCharm Community Edition

import numpy as np
import cv2
from PIL import Image
import pyprind
import time


def mask_process(path, n_classes=4, width=512,height=512):
    '''
    reshape original mask to (0,1,2,3)
    '''
    mask = np.zeros((height, width, n_classes))
    img = cv2.imread(path, 1)
    img = cv2.resize(img, (width, height))
    img = img[:, :, 0] + img[:, :, 1]
    #  127:bg, 0:lv, 229:mro, 211:rv
    pixel = {0: 127, 1: 0, 2: 229, 3: 211}
    for c in range(n_classes):
        mask[:, :, c] = (img == pixel[c]).astype(int)
    return mask


def export_all_contours(path, contours, crop_size, image_shape=512, n_classes=4):
    '''
    index images and masks
    '''
    print('\nProcessing {:d} images and labels ...\n'.format(len(contours)))
    images = np.zeros((len(contours), crop_size, crop_size, 3))
    masks = np.zeros((len(contours), crop_size, crop_size, n_classes))
    #   jindutiao
    bar = pyprind.ProgBar(len(contours), monitor=True, title="Processing images and labels")

    for idx, contour in enumerate(contours):
        str_contour = ' '.join(contour)
        img_file = str_contour.split(' ')[0]
        mask_file = str_contour.split(' ')[1]
        img = np.array(Image.open(path+img_file))
        #mask = np.array(Image.open(acdc_dir+mask_file))
        mask = mask_process(path+mask_file)
        img = np.reshape(img, (1,image_shape,image_shape,3))
        mask = np.reshape(mask, (1,image_shape,image_shape,n_classes))
        # img = center_crop(img, crop_size=crop_size)
        # mask = center_crop(mask, crop_size=crop_size)
        images[idx] = img
        masks[idx] = mask
        time.sleep(0.5)
        bar.update()
    print(bar)
    print('Done!')
    return images, masks