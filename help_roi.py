# *.* coding:utf-8 *.*
# @Time    : 18-6-8 上午11:03
# @Author  : Lxy
# @File    : help_roi.py
# @Software: PyCharm Community Edition

import numpy as np
import cv2
from PIL import Image

def get_roi(image, contour, shape_out=32):
    X_location = []
    Y_location = []
    for y in range(len(contour)):
        for x in range(len(contour[y])):
            #   miccai:
            #   if contour[y][x] == 215:
            #   acdc:
            #   if contour[y][x] != 14:
            if contour[y][x] != 14:
                X_location.append(x)
                Y_location.append(y)
    if X_location != [] and Y_location != []:
        X_min, Y_min = np.min(X_location), np.min(Y_location)
        X_max, Y_max = np.max(X_location), np.max(Y_location)
        w = X_max - X_min
        h = Y_max - Y_min
        mask_roi = np.zeros(image.shape)
        if w > h:
            mask_roi[int(Y_min - (w - h) // 2):int(Y_max +
                                                   (w - h) // 2), int(X_min):int(X_max)] = 1.0
        else:
            mask_roi[int(Y_min):int(Y_max), int(X_min - (h - w) // 2)
                         :int(X_max + (h - w) // 2)] = 1.0
        return cv2.resize(mask_roi, (shape_out, shape_out),
                          interpolation=cv2.INTER_NEAREST)
    else:
        return None


def create_acdc_dataset(acdc_path, re_size=64):
    f = open(acdc_path, 'r')
    img_contour = []
    imgs = []
    contours = []
    for line in f.readlines():
        img_contour.append([line[:-1]])
    for idx, contour in enumerate(img_contour):
        str_contour = ' '.join(contour)
        img_file = str_contour.split(' ')[0]
        mask_file = str_contour.split(' ')[1]
        img = np.array(Image.open(img_file).convert('L'))
        img = cv2.resize(img, (re_size,re_size))
        Contour = np.array(Image.open(mask_file).convert('L'))
        Contour = cv2.resize(Contour, (re_size,re_size))
        roi = get_roi(img, Contour)
        if roi is not None:
            imgs.append(img)
            contours.append(roi)
    return imgs, contours


def create_miccai_dataset(miccai_path='D:\data_256_on', miccai_label_path='D:\data_256_on_label', re_size=64):
    #   img 和 contour的地址
    img_list = []
    Contour_list = []
    #   resize之后的imgs和contours的数组
    imgs = []
    contours = []

    for i in range(279):
        img_list.append(miccai_path+'\pic_' + str(i) + '.png')
        Contour_list.append(
            miccai_label_path+'\pic_label_' +
            str(i) +
            '.png')
    for i in range(279):
        img = np.array(Image.open(img_list[i]).convert('L'))
        img = cv2.resize(img, (re_size,re_size))

        Contour = np.array(Image.open(Contour_list[i]).convert('L'))
        Contour = cv2.resize(Contour, (re_size,re_size))

        imgs.append(img)
        contours.append(get_roi(img, Contour))
    return imgs, contours

def computer_roi(y, image_size, roi_shape=32):
    pred = cv2.resize(y.reshape(roi_shape, roi_shape),
                      (image_size, image_size), cv2.INTER_NEAREST)
    pos_pred = np.array(np.where(pred > 0.5))
    # get the center of the mask
    X_min, Y_min = pos_pred[0, :].min(), pos_pred[1, :].min()
    X_max, Y_max = pos_pred[0, :].max(), pos_pred[1, :].max()
    X_middle = X_min + (X_max - X_min) / 2
    Y_middle = Y_min + (Y_max - Y_min) / 2
    # Find ROI coordinates
    X_top = int(X_middle - 50)
    Y_top = int(Y_middle - 50)
    X_down = int(X_middle + 50)
    Y_down = int(Y_middle + 50)
    # crop ROI of size 100x100
    mask_roi = np.zeros((image_size, image_size))
    mask_roi = cv2.rectangle(mask_roi, (X_top, Y_top),
                             (X_down, Y_down), 1, -1) * 255
    return mask_roi

def computer_roi_pred(y_pred, index, image_size,roi_shape=32):
    pred = cv2.resize(y_pred[index].reshape(roi_shape, roi_shape),
                      (image_size, image_size), cv2.INTER_NEAREST)
    pos_pred = np.array(np.where(pred > 0.5))
    # get the center of the mask
    X_min, Y_min = pos_pred[0, :].min(), pos_pred[1, :].min()
    X_max, Y_max = pos_pred[0, :].max(), pos_pred[1, :].max()
    X_middle = X_min + (X_max - X_min) / 2
    Y_middle = Y_min + (Y_max - Y_min) / 2
    # Find ROI coordinates
    X_top = int(X_middle - 50)
    Y_top = int(Y_middle - 50)
    X_down = int(X_middle + 50)
    Y_down = int(Y_middle + 50)
    # crop ROI of size 100x100
    mask_roi = np.zeros((image_size, image_size))
    mask_roi = cv2.rectangle(mask_roi, (X_top, Y_top),
                             (X_down, Y_down), 1, -1) * 255
    return mask_roi


def dice_acc(m,img_size,acdc_path):

    dic_mask = []
    dic_midd = []
    s = s_ground = s_midd = 0
    mask_ground = mask_midd = 0

    f = open(acdc_path, 'r')
    img_contour = []
    k = 0
    for line in f.readlines():
        img_contour.append([line[:-1]])
    for idx, contour in enumerate(img_contour):
        str_contour = ' '.join(contour)
        img_file = str_contour.split(' ')[0]
        mask_file = str_contour.split(' ')[1]
        x = np.array(Image.open(img_file).convert('L'))
        y = np.array(Image.open(mask_file).convert('L'))
        print(mask_file)
        img_x = cv2.resize(x, (64, 64))
        img_y = cv2.resize(y, (64,64))
        ground_truth = get_roi(img_x, img_y)
        if ground_truth is not None:
            k += 1
            ground_truth = computer_roi(ground_truth, img_size)
            pre = np.reshape(img_x, [1,64,64,1])
            y_pred = m.predict(pre, batch_size=1)
            mask_roi = computer_roi_pred(y_pred, 0, img_size)

            midd = np.zeros((img_size, img_size))
            midd = cv2.rectangle(midd, (img_size/2-50, img_size/2-50),
                                     (img_size/2+50, img_size/2+50), 1, -1) * 255


            for i in range(img_size):
                for j in range(img_size):
                    a = ground_truth[i, j]
                    b = mask_roi[i, j]
                    c = midd[i,j]
                    if b > 68:
                        s_ground = s_ground + 1
                    if c > 68:
                        s_midd += 1
                    if a > 68:
                        s = s + 1
                        if b > 68:
                            mask_ground+= 1
                        if c > 68:
                            mask_midd += 1
            res_mask = (2 * float(mask_ground)) / (float(s_ground) + float(s))
            res_midd = (2 * float(mask_midd)) / (float(s_midd) + float(s))
            dic_mask.append(res_mask)
            dic_midd.append(res_midd)
    print('DICE of mask:', np.mean(dic_mask))
    print('DICE of middle:', np.mean(dic_midd))
    print('Sum of Val: %s, Sum of ROI: %s\n'%(len(img_contour),k))