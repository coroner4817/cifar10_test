
# coding: utf-8

# In[26]:

import numpy as np
import os
import glob
import cv2
import math
import pickle
import datetime
import pdb;
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import model_from_json
from keras.initializations import normal, identity
from keras.layers.recurrent import SimpleRNN
from numpy.random import permutation


def VGG_16(weights_path, inputShape, nb_classes):
    #inputShape 3dim
    
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=inputShape))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

#     if weights_path:
#         model.load_weights(weights_path)

    model.summary()
    
    return model


def moustafa_model1(inputShape, nb_classes):
    model = Sequential()

    model.add(Convolution2D(62, 3, 3, border_mode='same', input_shape=inputShape))
    model.add(Activation('relu'))
    model.add(Convolution2D(62, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.summary()

    return model


def cifar10_cnn_model(inputShape, nb_classes):
    #inputShape 3dim
    
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=inputShape))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
              
    model.summary()
    
    return model


def cifar10_cnn_model_yingnan(inputShape, nb_classes):
    #inputShape 3dim
    
    model = Sequential()

    model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=inputShape))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
              
    model.summary()
    
    return model


def mnist_cnn_model(inputShape, nb_classes):
    #inputShape 3dim
    model = Sequential()

    # each input is 1*28*28, output is 32*26*26 because stride is 1 and image size is 28
    model.add(Convolution2D(32, 3, 3,
                            border_mode='valid',
                            input_shape=inputShape))
    model.add(Activation('relu'))
    # output is 32*24*24 because stride is 1 and input is 32*26*26
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    # pooling will reduce the output size to 32*12*12
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # dropout not effect the output shape
    model.add(Dropout(0.25))

    # conver the 32*12*12 output to 4608
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.summary()
              
    return model


# In[67]:

def mnist_mlp_model(inputShape, nb_classes):
    # inputShape xdim
    # init model layers
    model = Sequential()
    # the first layer need to know the input shape
    # dense is the fully connected NN layer (not CNN), 512 is the output dim
    # here is like have 512 neurons
    model.add(Dense(512, input_shape=inputShape))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    # print the summary description of the model
    model.summary()
    
    return model


# In[68]:

def mnist_irnn_model(inputShape, nb_classes):
    # inputShape 2dim
    model = Sequential()
    model.add(SimpleRNN(output_dim=100,
                        init=lambda shape, name: normal(shape, scale=0.001, name=name),
                        inner_init=lambda shape, name: identity(shape, scale=1.0, name=name),
                        activation='relu',
                        input_shape=inputShape))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.summary()


# In[69]:

def mnist_transferCNN_model(inputShape, nb_classes):
    # inputShape 3dim
    # define two groups of layers: feature (convolutions) and classification (dense)
    feature_layers = [
        Convolution2D(32, 3, 3,
                      border_mode='valid',
                      input_shape=inputShape),
        Activation('relu'),
        Convolution2D(32, 3, 3),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
    ]
    classification_layers = [
        Dense(128),
        Activation('relu'),
        Dropout(0.5),
        Dense(nb_classes),
        Activation('softmax')
    ]

    # create complete model
    model = Sequential()
    for l in feature_layers + classification_layers:
        model.add(l)
        
    model.summary()
    
    return model





