#!/usr/bin/env python2.7

from keras import optimizers
from keras.models import Model
from keras.layers import Dropout, Lambda
from keras.layers import Input, average, concatenate
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import ZeroPadding2D, Cropping2D
from keras.layers import UpSampling2D
from keras import backend as K


def mvn(tensor):
    '''Performs per-channel spatial mean-variance normalization.'''
    epsilon = 1e-6
    mean = K.mean(tensor, axis=(1,2), keepdims=True)
    std = K.std(tensor, axis=(1,2), keepdims=True)
    mvn = (tensor - mean) / (std + epsilon)
    
    return mvn


def crop(tensors):
    '''
    List of 2 tensors, the second tensor having larger spatial dimensions.
    '''
    h_dims, w_dims = [], []
    for t in tensors:
        b, h, w, d = K.get_variable_shape(t)
        h_dims.append(h)
        w_dims.append(w)
    crop_h, crop_w = (h_dims[1] - h_dims[0]), (w_dims[1] - w_dims[0])
    rem_h = crop_h % 2
    rem_w = crop_w % 2
    crop_h_dims = (crop_h / 2, crop_h / 2 + rem_h)
    crop_w_dims = (crop_w / 2, crop_w / 2 + rem_w)
    cropped = Cropping2D(cropping=(crop_h_dims, crop_w_dims))(tensors[1])
    
    return cropped


def dice_coef(y_true, y_pred, smooth=0.0):
    '''Average dice coefficient per batch.'''
    axes = (1,2,3)
    intersection = K.sum(y_true * y_pred, axis=axes)
    summation = K.sum(y_true, axis=axes) + K.sum(y_pred, axis=axes)
    
    return K.mean((2.0 * intersection + smooth) / (summation + smooth), axis=0)


def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred, smooth=10.0)
'''
def balance_loss(y_pred):
    pred = K.sum(y_pred,axis = (1,2,3))

    weight = 1-K.mean(pred,axis=0)/9216'''

def jaccard_coef(y_true, y_pred, smooth=0.0):
    '''Average jaccard coefficient per batch.'''
    axes = (1,2,3)
    intersection = K.sum(y_true * y_pred, axis=axes)
    union = K.sum(y_true, axis=axes) + K.sum(y_pred, axis=axes) - intersection
    return K.mean( (intersection + smooth) / (union + smooth), axis=0)

def fcn_model(input_shape, num_classes, weights=None):

    if num_classes == 2:
        num_classes = 1
        loss = dice_coef_loss
        activation = 'sigmoid'
    else:
        loss = 'categorical_crossentropy'
        activation = 'softmax'

    kwargs = dict(
        kernel_size=3,
        strides=1,
        activation='relu',
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
    )
    
    data = Input(shape=input_shape, dtype='float', name='data')
    mvn0 = Lambda(mvn, name='mvn0')(data)
    print(mvn0.shape)
    conv1 = Conv2D(filters=64, name='conv1', **kwargs)(mvn0)
    mvn1 = Lambda(mvn, name='mvn1')(conv1)
    print(mvn1.shape)
    conv2 = Conv2D(filters=64, name='conv2', **kwargs)(mvn1)
    mvn2 = Lambda(mvn, name='mvn2')(conv2)
    print(mvn2.shape)
    conv3 = Conv2D(filters=64, name='conv3', **kwargs)(mvn2)
    mvn3 = Lambda(mvn, name='mvn3')(conv3)
    print(mvn3.shape)
    pool1 = MaxPooling2D(pool_size=(2,2),name='pool1')(mvn3)
    drop1 = Dropout(rate=0.5, name='drop1')(pool1)

    
    conv4 = Conv2D(filters=128, name='conv4', **kwargs)(drop1)
    mvn4 = Lambda(mvn, name='mvn4')(conv4)
    print(mvn4.shape)
    conv5 = Conv2D(filters=128, name='conv5', **kwargs)(mvn4)
    mvn5 = Lambda(mvn, name='mvn5')(conv5)
    print(mvn5.shape)
    conv6 = Conv2D(filters=128, name='conv6', **kwargs)(mvn5)
    mvn6 = Lambda(mvn, name='mvn6')(conv6)
    print(mvn6.shape)
    conv7 = Conv2D(filters=128, name='conv7', **kwargs)(mvn6)
    mvn7 = Lambda(mvn, name='mvn7')(conv7)
    print(mvn7.shape)
    pool2 = MaxPooling2D(pool_size=(2,2),name='pool2')(mvn7)
    drop2 = Dropout(rate=0.5, name='drop2')(pool2)


    conv8 = Conv2D(filters=256, name='conv8', **kwargs)(drop2)
    mvn8 = Lambda(mvn, name='mvn8')(conv8)
    print(mvn8.shape)
    conv9 = Conv2D(filters=256, name='conv9', **kwargs)(mvn8)
    mvn9 = Lambda(mvn, name='mvn9')(conv9)
    print(mvn9.shape)
    conv10 = Conv2D(filters=256, name='conv10', **kwargs)(mvn9)
    mvn10 = Lambda(mvn, name='mvn10')(conv10)
    print(mvn10.shape)
    conv11 = Conv2D(filters=256, name='conv11', **kwargs)(mvn10)
    mvn11 = Lambda(mvn, name='mvn11')(conv11)
    print(mvn11.shape)
    pool3 = MaxPooling2D(pool_size=(2,2), name='pool3')(mvn11)
    drop3 = Dropout(rate=0.5, name='drop3')(pool3)


    conv12 = Conv2D(filters=512, name='conv12', **kwargs)(drop3)
    mvn12 = Lambda(mvn, name='mvn12')(conv12)
    print(mvn12.shape)
    conv13 = Conv2D(filters=512, name='conv13', **kwargs)(mvn12)
    mvn13 = Lambda(mvn, name='mvn13')(conv13)
    print(mvn13.shape)
    conv14 = Conv2D(filters=512, name='conv14', **kwargs)(mvn13)
    mvn14 = Lambda(mvn, name='mvn14')(conv14)
    print(mvn14.shape)
    conv15 = Conv2D(filters=512, name='conv15', **kwargs)(mvn14)
    mvn15 = Lambda(mvn, name='mvn15')(conv15)
    print(mvn15.shape)
    pool4 = MaxPooling2D(pool_size=(2,2), name='pool4')(mvn15)
    drop4 = Dropout(rate=0.5, name='drop4')(pool4)


    conv16 = Conv2D(filters=1024, name='conv16', **kwargs)(drop4)
    mvn16= Lambda(mvn, name='mvn16')(conv16)
    print(mvn16.shape)
    conv17 = Conv2D(filters=1024, name='conv17', **kwargs)(mvn16)
    mvn17 = Lambda(mvn, name='mvn17')(conv17)
    print(mvn17.shape)
    conv18 = Conv2D(filters=1024, name='conv18', **kwargs)(mvn17)
    mvn18 = Lambda(mvn, name='mvn18')(conv18)
    print(mvn18.shape)
    conv19 = Conv2D(filters=1024, name='conv19', **kwargs)(mvn18)
    mvn19 = Lambda(mvn, name='mvn19')(conv19)


    score_conv19 = Conv2D(filters=512, kernel_size=1,
                        strides=1, activation='relu', padding='valid',
                        kernel_initializer='glorot_uniform', use_bias=True,
                        name='score_conv19')(mvn19)
    upsample0 = UpSampling2D(size=(2,2))(score_conv19)
    fuse_scores0  =concatenate([mvn15, upsample0], name = 'fuse_scores0')

    conv_1_0 = Conv2D(filters=512, kernel_size=1,
                        strides=1, activation='relu', padding='valid',
                        kernel_initializer='glorot_uniform', use_bias=True,
                        name='conv_1_0')(fuse_scores0)
    print('conv_1_0: ', conv_1_0.shape)
    conv_1_3_0 = Conv2D(filters=256, kernel_size=3,
                        strides=1, activation='relu', padding='same',
                        kernel_initializer='glorot_uniform', use_bias=True,
                        name='conv_1_3_0')(conv_1_0)
    print('conv_1_3_0: ', conv_1_3_0.shape)


    upsample1 = UpSampling2D(size=(2,2))(conv_1_3_0)
    print('upsample1: ', upsample1.shape)
    fuse_scores1  =concatenate([mvn11, upsample1], name = 'fuse_scores1')
    print('fuse_scores1: ',fuse_scores1.shape)

    conv_1_1 = Conv2D(filters=256, kernel_size=1,
                      strides=1, activation='relu', padding='valid',
                      kernel_initializer='glorot_uniform', use_bias=True,
                      name='conv_1_1')(fuse_scores1)
    print('conv_1_1: ', conv_1_1.shape)
    upsample2 = UpSampling2D(size=(2,2))(conv_1_1)

    print('upsample2: ', upsample2.shape)

    fuse_scores2 = concatenate([mvn7, upsample2], name = 'fuse_scores2')
    print('fuse_scores2: ',fuse_scores2.shape)

    conv_1_2 = Conv2D(filters=128, kernel_size=1,
                      strides=1, activation='relu', padding='valid',
                      kernel_initializer='glorot_uniform', use_bias=True,
                      name='conv_1_2')(fuse_scores2)
    print('conv_1_2: ', conv_1_2.shape)
    upsample3 = UpSampling2D(size=(2,2))(conv_1_2)
    '''
    upsample3 = Conv2DTranspose(filters=128, kernel_size=3,
                        strides=2, activation=None, padding='valid',
                        kernel_initializer='glorot_uniform', use_bias=False,
                        name='upsample3')(conv_1_2)'''
    print('upsample3: ',upsample3.shape)
    #crop3 = Lambda(crop, name='crop3')([data, upsample3])
    ###conv3
    score_conv3 = Conv2D(filters=128,kernel_size=1,
                         strides=1,activation='relu', padding='valid',
                         kernel_initializer='glorot_uniform', use_bias=True,
                         name='score_conv3')(mvn3)
    print('score_conv3: ', score_conv3.shape)
    #crop3 = Lambda(crop, name='crop3')([data,upsample3])
    #print('crop3: ', crop3.shape)
    fuse_scores3 = concatenate([mvn3, upsample3], name = 'fuse_scores3')
    print('fuse_scores3: ',fuse_scores3.shape)

    conv_1_3 = Conv2D(filters=64, kernel_size=1,
                      strides=1, activation='relu', padding='valid',
                      kernel_initializer='glorot_uniform', use_bias=True,
                      name='conv_1_3')(fuse_scores3)
    print('conv_1_3: ', conv_1_3.shape)


    '''
    upsample4 = Conv2DTranspose(filters=64, kernel_size=3,
                                strides=2, activation=None, padding='valid',
                                kernel_initializer='glorot_uniform', use_bias=False,
                                name='upsample4')(conv_1_3)'''
    #upsample4 = UpSampling2D(fuse_scores3)


    #crop4 = Lambda(crop, name='crop4')([data, upsample4])


    #############
    predictions = Conv2D(filters=num_classes, kernel_size=1,
                        strides=1, activation=activation, padding='valid',
                        kernel_initializer='glorot_uniform', use_bias=True,
                        name='predictions')(conv_1_3)
    print('predictions: ', predictions.shape)
    model = Model(inputs=data, outputs=predictions)
    if weights is not None:
        model.load_weights(weights)
    #sgd = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
    adam = optimizers.adam(lr=1e-5)
    model.compile(optimizer=adam, loss=loss,
                  metrics=['accuracy',  dice_coef, jaccard_coef])

    return model

#
# if __name__ == '__main__':
#     model = fcn_model((512, 512, 1), 2, weights=None)


