try:
    # --------Helper------------
    import preprocess as pp

    # --------General------------
    import cv2 as cv2
    import os
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

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
    from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
    from keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta, RMSprop
    print("-------------------Libraries Loaded Successfully!!!-------------------------------------------")
except:
    print("Library not Found ! ")
#----------------------------------Loading the train data------------------------------------------------------------------------

#Informarion
PATH = "D:/Major/Models/lwdct/segmented_frameset_resized/train/"
IMAGE_SIZE = 128


# Loading the dataset
dat,Y=pp.load_dataset(PATH , IMAGE_SIZE)

#Converting labels into one hot
labels=np.array(Y)
num_classes=6
Y = np_utils.to_categorical(labels, num_classes)#onehot class Labels

# Class Labels
cl=["HandShaking","Hugging","Kicking","Pointing","Punching","PUSHING"]


#----------------Split data into train test--------------------------------------------------------------

epochs =  20
batch_size = int(.25 * 480)
if batch_size > 128:
    batch_size = 128 # batch size caps at 128



x,y = dat,Y
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

print("Shape of train set and labels:",X_train.shape,y_train.shape)
print("Shape of test set and labels:",X_test.shape,y_test.shape)



x_train_samp = X_train.astype('float32')
x_test_samp = X_test.astype('float32')
x_train_samp /= 255
x_test_samp /= 255

#input shape
input_shape=X_train[0].shape
print("Input Shape of Each Image:",input_shape)


model = Sequential()
model.add(Conv2D(96, (11,11),padding='same', input_shape=(64,64,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (5,5), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(384, (3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(384, (3,3),padding='same'))

model.add(Conv2D(256, (3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(768))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(4096))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5,name='last_layer'))

model.add(Dense(6))
model.add(BatchNormalization())
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

model.fit(x_train_samp,y_train,batch_size=batch_size,epochs=epochs)

# define the layer for feature extraction
intermediate_layer = Model(inputs=model.input, outputs=model.get_layer('last_layer').output)

feature_engineered_train = intermediate_layer.predict(x_train_samp)
feature_engineered_train = pd.DataFrame(feature_engineered_train)

feature_engineered_test = intermediate_layer.predict(x_test_samp)
feature_engineered_test = pd.DataFrame(feature_engineered_test)


print(feature_engineered_train.shape)
print(feature_engineered_test.shape)



