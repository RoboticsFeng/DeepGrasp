import os
import time
import numpy as np
from PIL import Image

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.utils import plot_model


img_width, img_height = 20, 20 # don't change
nb_train_samples = len(os.listdir('data/positive_grasp/')) + len(os.listdir('data/negative_grasp/'))
epochs = 100
batch_size = 32

# build model
if K.image_data_format() == 'channels_first':
    input_shape = (1, img_width, img_height)
else:
    input_shape = (img_width, img_height, 1)
model = Sequential()
model.add(Conv2D(8, (3, 3), input_shape = input_shape, padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(16, (3, 3), padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss = 'binary_crossentropy',
              optimizer = 'rmsprop',
              metrics = ['accuracy'])
print(model.summary())

trainX = np.empty((nb_train_samples, img_width, img_height,1),dtype = 'float32')
trainY = np.empty((nb_train_samples), dtype = 'int')
for i in range(len(os.listdir('data/positive_grasp/'))):
    image_name = 'data/positive_grasp/' + str(i).zfill(4) + '.png'
    trainX[i,:,:,0] = np.transpose(np.asarray(Image.open(image_name), dtype = 'float32')) * (1.0/255)
    trainY[i] = int(1)

nb_positive = len(os.listdir('data/positive_grasp/'))
for j in range(len(os.listdir('data/negative_grasp/'))):
    image_name = 'data/negative_grasp/' + str(i).zfill(4) + '.png'
    trainX[nb_positive+j,:,:,0] = np.transpose(np.asarray(Image.open(image_name), dtype = 'float32')) * (1.0/255)
    trainY[nb_positive+j] = int(0)

# this is the augmentation configuration we will use for training
train_data = ImageDataGenerator().flow(trainX, trainY, batch_size = batch_size)



start = time.clock()
 
hist = model.fit_generator(
    train_data,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs,
    validation_data = None,
    validation_steps = None)

end=time.clock()

model.save_weights('weights/test_weights.h5')
print('Time comsumption: %f s'%(end-start))
