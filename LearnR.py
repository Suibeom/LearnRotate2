import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from keras.models import Sequential
from keras.datasets import mnist
import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from keras.layers import Dense, Dropout, InputLayer
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


def GPP(X, y, X_):
    gp = GaussianProcessRegressor().fit(X[:, np.newaxis], y)
    y, y_sig = gp.predict(X_[:, np.newaxis], return_std=True)
    y_upper = y + 2 * y_sig
    return X_[np.argmax(y_upper)]


model5 = Sequential([
    InputLayer(input_shape=(1, 28, 28)),
    keras.layers.Flatten(),
    keras.layers.LeakyReLU(alpha=0.1),
    Dense(500),
    keras.layers.LeakyReLU(alpha=0.1),
    Dense(290),
    keras.layers.LeakyReLU(alpha=0.1),
    Dense(10, activation='softmax')
])
model5.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=['accuracy'])


def load_and_normalize_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / 256.0
    x_train = x_train.reshape(60000, 1, 28, 28)
    x_test = x_test / 256.0
    x_test = x_test.reshape(10000, 1, 28, 28)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    return (x_train, y_train), (x_test, y_test)


def sixes_and_nines():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / 256.0
    x_train = x_train.reshape(60000, 1, 28, 28)
    x_test = x_test / 256.0
    x_test = x_test.reshape(10000, 1, 28, 28)
    sixes_or_nines_train_idx = (y_train == 6) | (y_train == 9)
    sixes_or_nines_test_idx = (y_test == 6) | (y_test == 9)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    x_train = x_train[sixes_or_nines_train_idx]
    x_test = x_test[sixes_or_nines_test_idx]
    y_train = y_train[sixes_or_nines_train_idx]
    y_test = y_test[sixes_or_nines_test_idx]
    return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = load_and_normalize_mnist()


def run_with_schedule(tsched, model):
    for i in range(len(tsched)):
        print(i)
        t = tsched[i]

        print(t * 360)
        datagen1 = ImageDataGenerator(rotation_range=int(360 * t), height_shift_range=int(5 * t),
                                      width_shift_range=int(5 * t),
                                      data_format='channels_first')
        datagen2 = ImageDataGenerator(rotation_range=int(15 * t), height_shift_range=int(2 * t),
                                      width_shift_range=int(2 * t),
                                      data_format='channels_first')
        datagen1_data = datagen1.flow(x_train, y_train, batch_size=1000)
        datagen2_data = datagen2.flow(x_train, y_train, batch_size=2000)
        batcher = BatchCombinedIterator(datagen1_data, datagen2_data)
        if False:
            for j in range(1, 17):
                plt.subplot(4, 4, j).imshow(datagen_data[0][0][j][0])
            plt.savefig(f'./diag/epoch{i:02}.png')

        hist1 = model.fit_generator(batcher, validation_data=(x_test, y_test),
                                    steps_per_epoch=20, epochs=1)
        tough_acc = hist1.history["val_acc"][0]
    return tough_acc


def run_with_mixture(r, t, model):
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
    for i in range(t):
        t = 1.0
        print(i)
        datagen1 = ImageDataGenerator(rotation_range=int(360 * t), height_shift_range=int(5 * t),
                                      width_shift_range=int(5 * t),
                                      data_format='channels_first')
        datagen2 = ImageDataGenerator(rotation_range=int(90 * t), height_shift_range=int(2 * t),
                                      width_shift_range=int(2 * t),
                                      data_format='channels_first')
        datagen1_data = datagen1.flow(x_train, y_train, batch_size=int(3000 * r))
        datagen2_data = datagen2.flow(x_train, y_train, batch_size=int(3000 * (1 - r)))
        batcher = BatchCombinedIterator(datagen1_data, datagen2_data)
        if False:
            for j in range(1, 17):
                plt.subplot(4, 4, j).imshow(datagen_data[0][0][j][0])
            plt.savefig(f'./diag/epoch{i:02}.png')

        hist1 = model5.fit_generator(batcher, validation_data=(x_test, y_test),
                                     steps_per_epoch=20, epochs=1)
        tough_acc = hist1.history["val_acc"][0]
    return tough_acc


def run_sixes_and_nines(model, epochs):
    (x_train, y_train), (x_test, y_test) = sixes_and_nines()
    datagen = ImageDataGenerator(rotation_range=360, height_shift_range=6,
                                 width_shift_range=6,
                                 data_format='channels_first')
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=3000), validation_data=(x_test, y_test),
              steps_per_epoch=20, epochs=epochs)


tsched = np.array([0.3, 0.2, 0.1, 0.0, 0.2, 0.4, 0.8, 1.0, 1.0, 1.0, 0.0, 0.1, 0.2, 0.3, 0.0,
                   0.0, 0.0, 0.1, 0.2, 0.3, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
results = np.zeros(30)
# for i in range(3,15):
#     tsched = np.zeros(30)
#     tsched[range(i)] = 1.0
#     results[i] = run_with_schedule(tsched)
#     for i in range(15):
#         print(f'Dropoff at {i:2}')
#         print(results[i])

# r_tries = np.zeros(30)
#
# for i in range(30):
#     if i == 0:
#         r = 0.2
#     print(f'Now trying; {r:3}')
#     results[i] = run_with_mixture(r, 15)
#     r_tries[i] = r
#     for j in range(i + 1):
#         print(r_tries[j])
#         print(results[j])
#     r = GPP(r_tries[range(i + 1)], results[range(i + 1)], np.linspace(0, 1, 1000))
# print("Done!")
