import os

from keras.callbacks import History

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from keras.models import Model, Sequential
from keras.layers import Input
from keras.layers import Flatten, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.datasets import mnist
import keras
import numpy as np
from keras.layers import Dense, Dropout, Activation, InputLayer
from keras.optimizers import Nadam, Adam
from keras.preprocessing.image import ImageDataGenerator

model5 = Sequential([
InputLayer(input_shape=(1,28,28)),
keras.layers.Flatten(),
Dropout(0.4),
keras.layers.LeakyReLU(alpha=0.1),
Dense(500),
keras.layers.LeakyReLU(alpha=0.1),
Dense(290),
keras.layers.LeakyReLU(alpha=0.1),
Dense(10,activation='softmax')
])
model5.compile(optimizer='nadam', loss='categorical_crossentropy',metrics=['accuracy'])
(x_train, y_train), (x_test,y_test) = mnist.load_data()
x_train = x_train/256.0
x_train = x_train.reshape(60000,1,28,28)
x_test = x_test/256.0
x_test = x_test.reshape(10000,1,28,28)
y_test = keras.utils.to_categorical(y_test, num_classes=10)
y_train = keras.utils.to_categorical(y_train, num_classes=10)
calbak = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=25, verbose=10, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
#
# datagen = ImageDataGenerator(rotation_range=0,height_shift_range=0,width_shift_range=0,data_format='channels_first')
# hist1 = model5.fit_generator(datagen.flow(x_train,y_train,batch_size=3000),validation_data=(x_test,y_test),steps_per_epoch=20,epochs=2, callbacks=[calbak])

for i in range(10):
    print(3*i)
    t = (i/7.5)**2/4
    print(t * 360)
    datagen = ImageDataGenerator(rotation_range=360*t, height_shift_range=5*t, width_shift_range=5*t,
                                 data_format='channels_first')
    hist1 = model5.fit_generator(datagen.flow(x_train, y_train, batch_size=3000), validation_data=(x_test, y_test),
                                 steps_per_epoch=20, epochs=1, callbacks=[calbak])
    tough_acc = hist1.history["val_acc"]
    print(3*i + 1)
    print(t*90)
    datagen = ImageDataGenerator(rotation_range=360 * t/4, height_shift_range=5 * t/4, width_shift_range=5 * t/4,
                                 data_format='channels_first')
    hist2 = model5.fit_generator(datagen.flow(x_train, y_train, batch_size=3000), validation_data=(x_test, y_test),
                         steps_per_epoch=20, epochs=1, callbacks=[calbak])
    moderate_acc = hist2.history["val_acc"]
    print(3*i + 2)
    datagen = ImageDataGenerator(rotation_range=5, height_shift_range=1, width_shift_range=1,
                                 data_format='channels_first')
    hist3 = model5.fit_generator(datagen.flow(x_train, y_train, batch_size=3000), validation_data=(x_test, y_test),
                         steps_per_epoch=20, epochs=1, callbacks=[calbak])
    easy_acc = hist3.history["val_acc"]