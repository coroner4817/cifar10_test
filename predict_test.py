
# coding: utf-8

# In[1]:

import numpy as np
import os
import glob
import cv2
import math
import pickle
import datetime
import pandas as pd
from keras.models import model_from_json


img_rows = 48
img_cols = 64
color_type = 3
use_cache = 1


# In[2]:

def get_im(path, img_rows, img_cols, color_type):
    # Load as grayscale
    if color_type == 1:
        img = cv2.imread(path, 0)
    elif color_type == 3:
        img = cv2.imread(path)
    # Reduce size
    resized = cv2.resize(img, (img_cols, img_rows))
    # mean_pixel = [103.939, 116.799, 123.68]
    # resized = resized.astype(np.float32, copy=False)

    # for c in range(3):
    #    resized[:, :, c] = resized[:, :, c] - mean_pixel[c]
    # resized = resized.transpose((2, 0, 1))
    # resized = np.expand_dims(img, axis=0)
    return resized


# In[3]:

def load_test(img_rows, img_cols, color_type):
    print('Read test images')
    path = os.path.join('.', 'data', 'test', '*.jpg')
    print (path)
    files = glob.glob(path)
    X_test = []
    X_test_id = []
    total = 0
    thr = math.floor(len(files)/10)
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im(fl, img_rows, img_cols, color_type)
        X_test.append(img)
        X_test_id.append(flbase)
        total += 1
        if total % thr == 0:
            print('Read {} images from {}'.format(total, len(files)))

    return X_test, X_test_id


# In[4]:

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


# In[5]:

# read previous stored pickle file
def restore_data(path):
    data = dict()
    if os.path.isfile(path):
        print('Restore data from pickle........')
        file = open(path, 'rb')
        data = pickle.load(file)
    return data


# In[6]:

def read_and_normalize_test_data(img_rows, img_cols, color_type, minus):

    cache_path = os.path.join('cache', 'test_r_' + str(img_rows) +
                              '_c_' + str(img_cols) + '_t_' +
                              str(color_type) + '.dat')

    if not os.path.isfile(cache_path) or use_cache == 0:
        test_data, test_id = load_test(img_rows, img_cols, color_type)
        cache_data((test_data, test_id), cache_path)
    else:
        print('Restore test from cache!')
        (test_data, test_id) = restore_data(cache_path)

    test_data = np.array(test_data, dtype=np.uint8)

    if color_type == 1:
        test_data = test_data.reshape(test_data.shape[0], color_type,
                                      img_rows, img_cols)
    else:
        test_data = test_data.transpose((0, 3, 1, 2))

    test_data = test_data.astype('float32')

    if minus:
        mean_pixel = [103.939, 116.779, 123.68]
        for c in range(color_type):
            test_data[:, c, :, :] = test_data[:, c, :, :] - mean_pixel[c]

    test_data /= 255
    print('Test shape:', test_data.shape)
    print(test_data.shape[0], ' test samples')
    return test_data, test_id


# In[7]:

def read_model(cross=''):
    json_name = 'architecture1' + cross + '.json'
    weight_name = 'model_weights1' + cross + '.h5'
    print json_name
    model = model_from_json(open(os.path.join('cache', json_name)).read())
    model.load_weights(os.path.join('cache', weight_name))
    return model


# In[8]:

def create_submission(predictions, test_id):
   result1 = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3',
                                                'c4', 'c5', 'c6', 'c7',
                                                'c8', 'c9'])
   result1.insert(0, 'img', pd.Series(test_id, index=result1.index))

   now = datetime.datetime.now()
   if not os.path.isdir('subm'):
       os.mkdir('subm')
   suffix = str(now.strftime("%Y-%m-%d-%H-%M"))
   sub_file = os.path.join('subm', 'submission_' + suffix + '.csv')
   result1.to_csv(sub_file, index=False)


# In[9]:

test_data, test_id = read_and_normalize_test_data(img_rows, img_cols, color_type, False)
test_data -= np.mean(test_data, axis=0)
test_data /= np.std(test_data, axis=0)

model = read_model('_cifar10')
model.summary()
test_prediction = model.predict(test_data, batch_size=128, verbose=1)
create_submission(test_prediction, test_id)
print ('finish!')


# In[ ]:




# In[ ]:
