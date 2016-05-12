from __future__ import print_function 
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
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils
from keras.models import model_from_json
from numpy.random import permutation
from aug_algo import aug_algo
from CNNmodelLib import cifar10_cnn_model

dataAug = aug_algo()

np.random.seed(2016)
use_cache = 1
color_type_global = 3
img_rows_global, img_cols_global = 48, 64
batch_size_global = 64
random_state_global = 20
epoch_global = 20


def get_im(path, img_rows, img_cols, color_type=1):
    # Load as grayscale
    if color_type == 1:
        img = cv2.imread(path, 0)
    elif color_type == 3:
        img = cv2.imread(path)
    # Reduce size
    resized = cv2.resize(img, (img_cols, img_rows))
    return resized


# create a dict which map the img name to the driver id
def get_driver_data():
    dr = dict()
    path = os.path.join('.', 'data', 'driver_imgs_list.csv')
    print('Read drivers data')
    f = open(path, 'r')
    line = f.readline()
    while (1):
        line = f.readline()
        if line == '':
            break
        arr = line.strip().split(',')
        dr[arr[2]] = arr[0]
    f.close()
    return dr


# read the train imgs and return a sorted list of unique driver id
def load_train(img_rows, img_cols, color_type=1, aug=True):
    X_train = []
    y_train = []
    driver_id = []

    # get the driver_id - img name dict
    driver_data = get_driver_data()

    if aug:
        print('Read train images')
        for j in range(10):
            print('Load folder c{}'.format(j))
            path = os.path.join('.', 'data', 'train', 'c' + str(j), '*.jpg')
            # use regex to get all the imgs in each folder
            files = glob.glob(path)
            for fl in files:
                flbase = os.path.basename(fl)
                # load the img, resize it to (224, 224)
                # color_type = 1 grayscale, 3 color
                img = get_im(fl, img_rows, img_cols, color_type)
                X_train.append(img)
                y_train.append(j)
                driver_id.append(driver_data[flbase])
                
                imgTemp = dataAug.horizShiftLeft(img, 0.05, 0.15)
                X_train.append(imgTemp)
                y_train.append(j)
                driver_id.append(driver_data[flbase])
                
                imgTemp = dataAug.horizShiftRight(img, 0.05, 0.15)
                X_train.append(imgTemp)
                y_train.append(j)
                driver_id.append(driver_data[flbase])
                
                imgTemp = dataAug.vertiShiftUp(img, 0.05, 0.15)
                X_train.append(imgTemp)
                y_train.append(j)
                driver_id.append(driver_data[flbase])
                
                imgTemp = dataAug.vertiShiftDown(img, 0.05, 0.15)
                X_train.append(imgTemp)
                y_train.append(j)
                driver_id.append(driver_data[flbase])
                
               # imgTemp = dataAug.rotatedCW(img, 0.1, 0.25, 1.2)
               # X_train.append(imgTemp)
               # y_train.append(j)
               # driver_id.append(driver_data[flbase])
               # 
               # imgTemp = dataAug.rotatedCCW(img, 0.1, 0.25, 1.2)
               # X_train.append(imgTemp)
               # y_train.append(j)
               # driver_id.append(driver_data[flbase])
               # 
               # imgTemp = dataAug.cropSkretch(img, 0.05, 0.15)
               # X_train.append(imgTemp)
               # y_train.append(j)
               # driver_id.append(driver_data[flbase])
    else:
        print('Read train images')
        for j in range(10):
            print('Load folder c{}'.format(j))
            path = os.path.join('.', 'data', 'train', 'c' + str(j), '*.jpg')
            # use regex to get all the imgs in each folder
            files = glob.glob(path)
            for fl in files:
                flbase = os.path.basename(fl)
                # load the img, resize it to (224, 224)
                # color_type = 1 grayscale, 3 color
                img = get_im(fl, img_rows, img_cols, color_type)
                X_train.append(img)
                y_train.append(j)
                driver_id.append(driver_data[flbase])
            
    # set(driver_id) to select only the unique driver id in the list
    # then convert back to list and sorted it
    unique_drivers = sorted(list(set(driver_id)))
    print('Unique drivers: {}'.format(len(unique_drivers)))
    return X_train, y_train, driver_id, unique_drivers


# write the data into a pickle file save in .dat format
def cache_data(data, path):
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    if os.path.isdir(os.path.dirname(path)):
        file = open(path, 'wb')
        pickle.dump(data, file)
        file.close()
    else:
        print('Directory doesnt exists')


# read previous stored pickle file
def restore_data(path):
    data = dict()
    if os.path.isfile(path):
        print('Restore data from pickle........')
        file = open(path, 'rb')
        data = pickle.load(file)
    return data


# save the model and the parameter
def save_model(model, index, cross=''):
    json_string = model.to_json()
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    json_name = 'architecture' + str(index) + cross + '.json'
    weight_name = 'model_weights' + str(index) + cross + '.h5'
    open(os.path.join('cache', json_name), 'w').write(json_string)
    model.save_weights(os.path.join('cache', weight_name), overwrite=True)


def read_model(index, cross=''):
    json_name = 'architecture' + str(index) + cross + '.json'
    weight_name = 'model_weights' + str(index) + cross + '.h5'
    model = model_from_json(open(os.path.join('cache', json_name)).read())
    model.load_weights(os.path.join('cache', weight_name))
    return model


def read_and_normalize_and_shuffle_train_data(img_rows, img_cols, color_type=1, aug=True):
    # cache folder to store the pkl file
    cache_path = os.path.join('cache', 'train_r_' + str(img_rows) + '_c_' + str(img_cols) + '_t_' + str(color_type) + '.dat')

    if not os.path.isfile(cache_path) or use_cache == 0:
        # initially load the train data into the pickle of the cahce folder
        train_data, train_target, driver_id, unique_drivers = load_train(img_rows, img_cols, color_type, aug)
        cache_data((train_data, train_target, driver_id, unique_drivers), cache_path)
    else:
        # direct load the pickle in the cache folder
        print('Restore train from cache!')
        (train_data, train_target, driver_id, unique_drivers) = restore_data(cache_path)
 
        
    # convert the data type to uint8 and convert to np array
    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)
    train_data = np.asarray(train_data)

    
    if color_type == 1:
        # reshape train_data to (count, channels=1, 224, 224)
        train_data = train_data.reshape(train_data.shape[0], color_type, img_rows, img_cols)
    else:
        train_data = train_data.transpose((0, 3, 1, 2))

       
    # convert class vectors to binary class matrices
    # convert the class number (3) to a vector like (0, 0, 0, 1, 0, 0,...)
    # the position 3 is 1 other is 0, this is for use with categorical_crossentropy
    train_target = np_utils.to_categorical(train_target, 10)

    # convert the data type to float32
    # train_data = train_data.astype('float32')

    # normalize
    # train_data /= 255

    return train_data, train_target, driver_id, unique_drivers


def copy_selected_drivers(train_data, train_target, driver_id, driver_list):
    data = []
    target = []
    index = []
    for i in range(len(driver_id)):
        if driver_id[i] in driver_list:
            data.append(train_data[i])
            target.append(train_target[i])
            index.append(i)
    data = np.array(data, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    index = np.array(index, dtype=np.uint32)
    return data, target, index   


def get_model(inputShape, nb_class):
    model = cifar10_cnn_model(inputShape, nb_class)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])
    
    return model
    

def run_cross_validation_full(nfolds=10, nb_epoch=10, split=0.2, modelStr=''):

    # Now it loads color image
    # input image dimensions
    img_rows, img_cols = img_rows_global, img_cols_global
    batch_size = batch_size_global
    
    train_data, train_label, driver_id, unique_drivers = read_and_normalize_and_shuffle_train_data(img_rows, img_cols, color_type_global, True)

    # shuffle the unique_drivers list
    unique_drivers = np.asarray(unique_drivers)
    perm = permutation(len(unique_drivers))
    unique_drivers = unique_drivers[perm]
    
    idxS = math.floor(split*len(unique_drivers))
    
    train_data_final = []
    train_label_final = []
    train_driverID_final = []
    
    valid_data_final = []
    valid_label_final = []
    valid_driverID_final = []
    
    for id in range(0, len(driver_id)):
        itemIdx = np.where(unique_drivers == driver_id[id])[0][0]
        if itemIdx<idxS:
            valid_data_final.append(train_data[id])
            valid_label_final.append(train_label[id])
            valid_driverID_final.append(driver_id[id])
        else:
            train_data_final.append(train_data[id])
            train_label_final.append(train_label[id])
            train_driverID_final.append(driver_id[id])
        
    
    valid_data_final = np.array(valid_data_final, dtype=np.float32)
    valid_label_final = np.array(valid_label_final, dtype=np.uint8)
    train_data_final = np.array(train_data_final, dtype=np.float32)
    train_label_final = np.array(train_label_final, dtype=np.uint8)
    
    valid_data_final -= np.mean(valid_data_final, axis=0)
    valid_data_final /= np.std(valid_data_final, axis=0)
    train_data_final -= np.mean(train_data_final, axis=0)
    train_data_final /= np.std(train_data_final, axis=0)
    
    print('Training Size: ', train_data_final.shape[0])
    print('Validating Size: ', valid_data_final.shape[0])
    
    
    model = get_model((color_type_global, img_rows, img_cols), 10)

    model.fit(train_data_final, train_label_final, 
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              verbose=1,
              validation_data=(valid_data_final, valid_label_final))
    

    save_model(model, 1, modelStr)
    
    
    

run_cross_validation_full(2, epoch_global, 0.15, '_cifar10_48_64_RMS_divide_dr')

