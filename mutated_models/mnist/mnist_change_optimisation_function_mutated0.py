
from __future__ import print_function
import keras, sys
from operators import activation_function_operators
from operators import training_data_operators
from operators import bias_operators
from operators import weights_operators
from operators import optimiser_operators
from operators import dropout_operators, hyperparams_operators
from operators import training_process_operators
from operators import loss_operators
from utils import mutation_utils
from utils import properties
from keras import optimizers
import os
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf

def main(model_name):
    model_location = os.path.join('trained_models', model_name)
    ((x_train, y_train), (x_test, y_test)) = mnist.load_data()
    (img_rows, img_cols) = (28, 28)
    num_classes = 10
    if (K.image_data_format() == 'channels_first'):
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    if (not os.path.exists(model_location)):
        batch_size = 128
        epochs = 12
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimiser_operators.operator_change_optimisation_function(keras.optimizers.Adadelta(learning_rate=1.0)), metrics=['accuracy'])
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
        model.save(os.path.join('trained_models', 'mnist_trained.h5'))
        score = model.evaluate(x_train, y_train, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        return score
    else:
        graph1 = tf.Graph()
        with graph1.as_default():
            session1 = tf.compat.v1.Session()
            with session1.as_default():
                model = tf.keras.models.load_model(model_location)
                score = model.evaluate(x_train, y_train, verbose=0)
                print(('score:' + str(score)))
        return score
if (__name__ == '__main__'):
    score = main('')
