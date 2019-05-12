# Written by Alexander Oshin
# References: Tensorflow and Keras documentation, MNIST documentation


import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model



if __name__ == '__main__':
    mnist = tf.keras.datasets.mnist
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    img_height, img_width = x_train[0].shape

    x_train = x_train.reshape((len(x_train), img_height, img_width, 1))
    x_test = x_test.reshape((len(x_test), img_height, img_width, 1))
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    input = Input(shape=(28, 28, 1))
    x = Conv2D(64, kernel_size=(3, 3))(input)
    x = Conv2D(32, kernel_size=(3, 3))(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(128)(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(input, output)
    model.compile(optimizer=tf.keras.optimizers.Adadelta(), loss=tf.keras.losses.categorical_crossentropy)

    model.fit(x_train, y_train, epochs=10)
    model.evaluate(x_test, y_test)

    model.save('classifier.h5')
    del model
