# -*- coding: utf-8 -*-
# @Time    : 2018/5/30 10:23
# @Author  : Lxy
# @Site    :
# @File    : CNN_ROI.py
# @Software: PyCharm

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras.losses import mean_squared_error
from keras.callbacks import ModelCheckpoint
import glob
import matplotlib.patches as patches
import json
import numpy as np
from matplotlib.path import Path
import dicom
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os
from help_roi import create_acdc_dataset, dice_acc

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


#   尝试使用Tensorflow进行
# def creat_model(image):
#
#     conv1 = tf.layers.conv2d(inputs=image, filters=100, kernel_size=11, strides=1,
#                              padding='valid', activation=tf.nn.relu)
#     pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2, padding='valid')
#     flat = tf.reshape(pool1, [-1, 9*9*100])
#     output = tf.layers.dense(flat, 1024, activation='sigmoid',
#                              kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0001))
#     output = tf.reshape(output, [-1, 32, 32])
#     return output

#   Keras
def create_model(input_shape=(64, 64)):

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', strides=(1, 1),
                     input_shape=(input_shape[0], input_shape[1], 1), kernel_initializer='uniform', activation='relu'))
    model.add(AveragePooling2D((2, 2)))

    model.add(Conv2D(64, (3,3), padding='same', strides=(1, 1),kernel_initializer='uniform', activation='relu'))
    model.add(AveragePooling2D((2, 2)))

    model.add(Conv2D(128, (3,3), padding='same', strides=(1, 1),kernel_initializer='uniform', activation='relu'))
    model.add(AveragePooling2D((2,2)))

    model.add(Reshape([-1, 8192]))
    model.add(
        Dense(
            1024,
            activation='sigmoid',
            kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Reshape([-1, 32, 32]))

    return model


def train(m, X, Y, callbacks, epochs,validation_split=0.33, batch_size=16, verbose=1, data_augm=False):
    '''
    Training the CNN to get ROI with data agumentation
    '''
    if data_augm:
        datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=50,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=False)
        datagen.fit(X)
        history = m.fit_generator(datagen.flow(X, Y, batch_size=batch_size),
                                  steps_per_epoch=X.shape[0] // batch_size,
                                  epochs=epochs)
    else:
        history = m.fit(X, Y, callbacks=callbacks, validation_split=validation_split, verbose=verbose,
                        batch_size=batch_size, epochs=epochs)

    return history


def main():
    m = create_model()
    m.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])
    print('Size for each layer :\nLayer, Input Size, Output Size')
    for p in m.layers:
        print(p.name.title(), p.input_shape, p.output_shape)

    acdc_path = 'train_acdc.txt'
    imgs, contours = create_acdc_dataset(acdc_path, re_size=64)
    print ('\n%s images for Dataset\n'%len(imgs))
    ratio = 0.8
    s = np.int(len(imgs) * ratio)
    img_train = imgs[:s]
    contour_train = contours[:s]

    img_val = imgs[s:]
    contour_val = contours[s:]

    #   create trainset and valset
    X_train = np.reshape(np.array(img_train), [len(img_train), 64, 64, 1])
    Y_train = np.reshape(
        np.array(contour_train), [
            len(contour_train), 1, 32, 32])

    X_val = np.reshape(np.array(img_val), [len(img_val), 64, 64, 1])
    Y_val = np.reshape(np.array(contour_val), [len(contour_val), 1, 32, 32])
    filepath = "weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                    save_weights_only=False, mode='min', period=1)
    callbacks_list = [checkpoint]
    h = train(m, X_train, Y_train, epochs=50,callbacks=callbacks_list,validation_split=0.33,
              batch_size=32, verbose=1,data_augm=False)

    #m.save('roi.h5')

    metric = 'loss'

    # plt.plot(range(len(h.history[metric])), h.history[metric])
    # plt.ylabel(metric)
    # plt.xlabel('epochs')
    # plt.title("Learning curve")
    # plt.savefig('Loss.png')
    # plt.show()

    # y_pred = m.predict(X_val, batch_size=16)
    #
    # mask_roi = computer_roi_pred(y_pred, 0)
    # plt.imshow(mask_roi)
    # plt.show()

def test():
    m = create_model()
    m.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])
    print('Size for each layer :\nLayer, Input Size, Output Size')
    for p in m.layers:
        print(p.name.title(), p.input_shape, p.output_shape)
    m.load_weights('weights.best.hdf5')
    dice_acc(m,512,'val_acdc_2.txt')


if __name__ == '__main__':
    test()