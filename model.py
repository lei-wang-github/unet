import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
import math
import configparser

config = configparser.ConfigParser()
config.read('configuration.txt')
img_height = int(config['data attributes']['image_height'])
img_width = int(config['data attributes']['image_width'])
model_reduction_ratio = int(config['model type']['modelReductionRatio'])
learningRate = float(config['training settings']['learningRate'])

def unet_original(pretrained_weights = None,input_size = (img_height,img_width,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = Adam(lr = learningRate), loss = 'binary_crossentropy', metrics = ['accuracy'])

    model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model


def unet(pretrained_weights=None, input_size=(img_height, img_width, 1)):
    reduction_ratio = model_reduction_ratio
    inputs = Input(input_size)
    conv1 = Conv2D(int(64/reduction_ratio), 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(int(64/reduction_ratio), 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(int(128/reduction_ratio), 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(int(128/reduction_ratio), 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(int(256/reduction_ratio), 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(int(256/reduction_ratio), 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(int(512/reduction_ratio), 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(int(512/reduction_ratio), 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(int(1024/reduction_ratio), 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(int(1024/reduction_ratio), 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(int(512/reduction_ratio), 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(int(512/reduction_ratio), 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(int(512/reduction_ratio), 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(int(256/reduction_ratio), 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(int(256/reduction_ratio), 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(int(256/reduction_ratio), 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(int(128/reduction_ratio), 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(int(128/reduction_ratio), 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(int(128/reduction_ratio), 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(int(64/reduction_ratio), 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(int(64/reduction_ratio), 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(int(64/reduction_ratio), 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    # model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=Adam(lr=learningRate), loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def unet_sepconv(pretrained_weights=None, input_size=(img_height, img_width, 1)):
    reduction_ratio = model_reduction_ratio
    inputs = Input(input_size)
    conv1 = SeparableConv2D(int(64/reduction_ratio), 3, activation='relu', padding='same', depthwise_initializer='he_normal', pointwise_initializer='he_normal')(inputs)
    conv1 = SeparableConv2D(int(64 / reduction_ratio), 3, activation='relu', padding='same', depthwise_initializer='he_normal', pointwise_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = SeparableConv2D(int(128/reduction_ratio), 3, activation='relu', padding='same', depthwise_initializer='he_normal', pointwise_initializer='he_normal')(pool1)
    conv2 = SeparableConv2D(int(128/reduction_ratio), 3, activation='relu', padding='same', depthwise_initializer='he_normal', pointwise_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = SeparableConv2D(int(256/reduction_ratio), 3, activation='relu', padding='same', depthwise_initializer='he_normal', pointwise_initializer='he_normal')(pool2)
    conv3 = SeparableConv2D(int(256/reduction_ratio), 3, activation='relu', padding='same', depthwise_initializer='he_normal', pointwise_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = SeparableConv2D(int(512/reduction_ratio), 3, activation='relu', padding='same', depthwise_initializer='he_normal', pointwise_initializer='he_normal')(pool3)
    conv4 = SeparableConv2D(int(512/reduction_ratio), 3, activation='relu', padding='same', depthwise_initializer='he_normal', pointwise_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = SeparableConv2D(int(1024/reduction_ratio), 3, activation='relu', padding='same', depthwise_initializer='he_normal', pointwise_initializer='he_normal')(pool4)
    conv5 = SeparableConv2D(int(1024/reduction_ratio), 3, activation='relu', padding='same', depthwise_initializer='he_normal', pointwise_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = SeparableConv2D(int(512/reduction_ratio), 2, activation='relu', padding='same', depthwise_initializer='he_normal', pointwise_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = SeparableConv2D(int(512/reduction_ratio), 3, activation='relu', padding='same', depthwise_initializer='he_normal', pointwise_initializer='he_normal')(merge6)
    conv6 = SeparableConv2D(int(512/reduction_ratio), 3, activation='relu', padding='same', depthwise_initializer='he_normal', pointwise_initializer='he_normal')(conv6)

    up7 = SeparableConv2D(int(256/reduction_ratio), 2, activation='relu', padding='same', depthwise_initializer='he_normal', pointwise_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = SeparableConv2D(int(256/reduction_ratio), 3, activation='relu', padding='same', depthwise_initializer='he_normal', pointwise_initializer='he_normal')(merge7)
    conv7 = SeparableConv2D(int(256/reduction_ratio), 3, activation='relu', padding='same', depthwise_initializer='he_normal', pointwise_initializer='he_normal')(conv7)

    up8 = SeparableConv2D(int(128/reduction_ratio), 2, activation='relu', padding='same', depthwise_initializer='he_normal', pointwise_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = SeparableConv2D(int(128/reduction_ratio), 3, activation='relu', padding='same', depthwise_initializer='he_normal', pointwise_initializer='he_normal')(merge8)
    conv8 = SeparableConv2D(int(128/reduction_ratio), 3, activation='relu', padding='same', depthwise_initializer='he_normal', pointwise_initializer='he_normal')(conv8)

    up9 = SeparableConv2D(int(64/reduction_ratio), 2, activation='relu', padding='same', depthwise_initializer='he_normal', pointwise_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = SeparableConv2D(int(64/reduction_ratio), 3, activation='relu', padding='same', depthwise_initializer='he_normal', pointwise_initializer='he_normal')(merge9)
    conv9 = SeparableConv2D(int(64/reduction_ratio), 3, activation='relu', padding='same', depthwise_initializer='he_normal', pointwise_initializer='he_normal')(conv9)
    conv9 = SeparableConv2D(2, 3, activation='relu', padding='same', depthwise_initializer='he_normal', pointwise_initializer='he_normal')(conv9)
    conv10 = SeparableConv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    # model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=Adam(lr=learningRate), loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)
    return model

# https://github.com/orobix/retina-unet/blob/master/src/retinaNN_training.py
def unetSmall (n_ch=1,patch_height=img_height,patch_width=img_width):
    inputs = Input(shape=(patch_height,patch_width, n_ch))
    reduction_ratio = model_reduction_ratio
    conv1 = Conv2D(math.ceil(int(32/reduction_ratio)), (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(math.ceil(int(32/reduction_ratio)), (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    #
    conv2 = Conv2D(int(64/reduction_ratio), (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(int(64/reduction_ratio), (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    #
    conv3 = Conv2D(int(128/reduction_ratio), (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(int(128/reduction_ratio), (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv3)

    up1 = UpSampling2D(size=(2, 2))(conv3)
    up1 = concatenate([conv2,up1],axis=3)
    conv4 = Conv2D(int(64/reduction_ratio), (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(int(64/reduction_ratio), (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    #
    up2 = UpSampling2D(size=(2, 2))(conv4)
    up2 = concatenate([conv1,up2], axis=3)
    conv5 = Conv2D(math.ceil(int(32/reduction_ratio)), (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(math.ceil(int(32/reduction_ratio)), (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    #
    conv6 = Conv2D(2, (1, 1), activation='relu',padding='same', kernel_initializer='he_normal')(conv5)
    conv7 = Conv2D(1, 1, activation='sigmoid', kernel_initializer='he_normal')(conv6)

    model = Model(inputs=inputs, outputs=conv7)

    model.compile(optimizer=Adam(lr=learningRate), loss='binary_crossentropy', metrics=['accuracy'])
    
    model.summary()
    
    return model

