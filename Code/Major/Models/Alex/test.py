try:
    # --------Helper------------
    import preprocess as pp
    # --------General------------
    import cv2 as cv2
    import os
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt

    # ----sklearn----------------
    from sklearn.utils import shuffle
    from sklearn.model_selection import train_test_split

    from keras.utils import np_utils
    # --------Tensorflow---------
    import tensorflow as tf


    # --------Keras--------------
    import keras
    from keras import backend as K
    # K.set_image_dim_ordering('th')
    from keras.callbacks import ModelCheckpoint, EarlyStopping
    from keras import layers, callbacks
    from keras.models import Model
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPool2D

    from keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta, RMSprop

    print("Libraries Loaded Successfully!!!")
except:
    print("Library not Found ! ")


pathi="D:/Major/segmented_set/validation/"
patho="D:/Major/Models/Alex/segmented_frameset_resized/validate/"
#Generates 20 frames for each video in validationset
#pp.frames_20(20,pathi,patho)


#Store in matrix
dat,l=pp.load_test(patho)


#Converting Matrix into an np array

data=np.array(dat)
labels=np.array(l)
num_classes=6
vY = np_utils.to_categorical(labels, num_classes)#onehot class Labels

