import os

from keras.callbacks import History

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from keras.models import Model, Sequential
from keras.layers import Input
from keras.layers import Flatten, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.datasets import mnist
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Activation, InputLayer
from keras.optimizers import Nadam, Adam
from keras.preprocessing.image import ImageDataGenerator


class BatchCombinedIterator():

    def __init__(self, a, b):
        self.a = a
        self.b = b
        a0, a1 = next(self.a)
        b0, b1 = next(self.b)
        self.current = np.concatenate((a0, b0)), np.concatenate((a1, b1))

    def __iter__(self):
        return self

    def __next__(self):
        a0, a1 = next(self.a)
        b0, b1 = next(self.b)
        self.current = np.concatenate((a0, b0)), np.concatenate((a1, b1))
        return self.current


model5 = Sequential([
    InputLayer(input_shape=(1, 28, 28)),
    keras.layers.Flatten(),
    Dropout(0.4),
    keras.layers.LeakyReLU(alpha=0.1),
    Dense(500),
    keras.layers.LeakyReLU(alpha=0.1),
    Dense(290),
    keras.layers.LeakyReLU(alpha=0.1),
    Dense(10, activation='softmax')
])
model5.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=['accuracy'])
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 256.0
x_train = x_train.reshape(60000, 1, 28, 28)
x_test = x_test / 256.0
x_test = x_test.reshape(10000, 1, 28, 28)
y_test = keras.utils.to_categorical(y_test, num_classes=10)
y_train = keras.utils.to_categorical(y_train, num_classes=10)
calbak = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=25, verbose=10, mode='auto',
                                           min_delta=0.0001, cooldown=0, min_lr=0)
#
# datagen = ImageDataGenerator(rotation_range=0,height_shift_range=0,width_shift_range=0,data_format='channels_first')
# hist1 = model5.fit_generator(datagen.flow(x_train,y_train,batch_size=3000),validation_data=(x_test,y_test),steps_per_epoch=20,epochs=2, callbacks=[calbak])

def run_with_schedule(tsched):
    model5 = Sequential([
        InputLayer(input_shape=(1, 28, 28)),
        keras.layers.Flatten(),
        Dropout(0.4),
        keras.layers.LeakyReLU(alpha=0.1),
        Dense(500),
        keras.layers.LeakyReLU(alpha=0.1),
        Dense(290),
        keras.layers.LeakyReLU(alpha=0.1),
        Dense(10, activation='softmax')
    ])
    model5.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=['accuracy'])
    for i in range(len(tsched)):
        print(i)
        t = tsched[i]

        print(t * 360)
        datagen1 = ImageDataGenerator(rotation_range=int(360*t), height_shift_range=int(5*t), width_shift_range=int(5*t),
                                     data_format='channels_first')
        datagen2 = ImageDataGenerator(rotation_range=int(15 * t), height_shift_range=int(2 * t),
                                     width_shift_range=int(2 * t),
                                     data_format='channels_first')
        datagen1_data = datagen1.flow(x_train, y_train, batch_size=1000)
        datagen2_data = datagen2.flow(x_train, y_train, batch_size=2000)
        batcher = BatchCombinedIterator(datagen1_data, datagen2_data)
        if False:
            for j in range(1,17):
                plt.subplot(4,4,j).imshow(datagen_data[0][0][j][0])
            plt.savefig(f'./diag/epoch{i:02}.png')

        hist1 = model5.fit_generator(batcher, validation_data=(x_test, y_test),
                                     steps_per_epoch=20, epochs=1, callbacks=[calbak])
        tough_acc = hist1.history["val_acc"][0]
    return tough_acc


tsched = np.array([0.3, 0.2, 0.1, 0.0, 0.2, 0.4, 0.8, 1.0, 1.0, 1.0, 0.0, 0.1, 0.2, 0.3, 0.0,
                   0.0, 0.0, 0.1, 0.2, 0.3, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
results = np.zeros(15)
for i in range(3,15):
    tsched = np.zeros(30)
    tsched[range(i)] = 1.0
    results[i] = run_with_schedule(tsched)
    for i in range(15):
        print(f'Dropoff at {i:2}')
        print(results[i])



