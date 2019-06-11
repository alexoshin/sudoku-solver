# Written by Alexander Oshin
# References: Tensorflow and Keras documentation, MNIST documentation
# Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG): https://arxiv.org/abs/1409.1556


import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model


def train_classifier(data_dir, save_dir, plot_loss=False):

    print('Training classifier...')

    num_classes = 9

    with open(data_dir, 'rb') as f:
        data = pickle.load(f)
    images = data['images']
    img_height, img_width = images[0].shape
    images = images.reshape((len(images), img_height, img_width, 1))
    labels = data['labels'] - 1
    labels = tf.keras.utils.to_categorical(labels, num_classes)
    indices = np.random.permutation(len(images))
    split_index = int(len(indices) * 0.9)
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    x_train = images[train_indices]
    y_train = labels[train_indices]
    x_test = images[test_indices]
    y_test = labels[test_indices]

    input = Input(shape=(28, 28, 1))
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(input)
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(input, output)
    model.compile(optimizer=tf.keras.optimizers.Adadelta(), loss=tf.keras.losses.categorical_crossentropy)

    history = model.fit(x_train, y_train, epochs=25, verbose=1)
    if plot_loss:
        plt.plot(history.history['loss'])
        plt.title('Loss over time')
        plt.show()
    model.evaluate(x_test, y_test)

    model.save(os.path.join(save_dir, 'classifier.h5'))
    del model
    print('Classifier saved to file.')


if __name__ == '__main__':
    train_classifier('./data/font_data_augmented.pickle', './data')
