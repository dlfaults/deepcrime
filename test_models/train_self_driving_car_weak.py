import argparse
import os

import numpy as np
import pandas as pd
import tensorflow
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Lambda, Conv2D, Dropout, Dense, Flatten, Convolution2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras import backend as K
from test_models.utils import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS

import logging
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# import warnings filter
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
from sklearn.utils import shuffle

from test_models.batch_generator import Generator
from test_models.utils import INPUT_SHAPE

np.random.seed(0)

def load_data(args):
    """
    Load training data and split it into training and validation set
    """
    tracks = ["track1"]
    drive = ['normal', 'recovery', 'reverse']

    x = None
    y = None
    path = None
    x_train = None
    y_train = None
    x_valid = None
    y_valid = None

    for track in tracks:
        for drive_style in drive:
            try:
                path = os.path.join(args.data_dir, track, drive_style, 'driving_log.csv')
                data_df = pd.read_csv(path)
                if x is None:
                    x = data_df[['center', 'left', 'right']].values
                    y = data_df['steering'].values
                else:
                    x = np.concatenate((x, data_df[['center', 'left', 'right']].values), axis=0)
                    y = np.concatenate((y, data_df['steering'].values), axis=0)
            except FileNotFoundError:
                print("Unable to read file %s" % path)
                continue

    if x is None or y is None:
        print("No driving data were provided for training. Provide correct paths to the driving_log.csv files")
        exit()

    try:
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=args.test_size, random_state=0)
    except TypeError:
        print("Missing header to csv files")
        exit()

    print("Train dataset: " + str(len(x_train)) + " elements")
    print("Test dataset: " + str(len(x_valid)) + " elements")
    return x_train, x_valid, y_train, y_valid


def build_model(args):
    """
    Modified NVIDIA model
    """
    model = Sequential()
    model.add(Lambda((lambda x: ((x / 127.5) - 1.0)), input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Dropout(args.keep_prob))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))

    model.summary()
    return model


def get_generators(args, x_train, x_valid, y_train, y_valid):
    x_train, y_train = shuffle(x_train, y_train, random_state=0)
    x_valid, y_valid = shuffle(x_valid, y_valid, random_state=0)

    x_train: 'x_train'
    y_train: 'y_train'

    train_generator = Generator(x_train, y_train, True, args)
    validation_generator = Generator(x_valid, y_valid, False, args)

    return train_generator, validation_generator


def train_model(model_dir, model, args, x_train, x_valid, y_train, y_valid):

    checkpoint = ModelCheckpoint('EPOCH' + '-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=args.save_best_only,  # save the model only if the val_loss gets low
                                 mode='auto',
                                 period=20)
    print(args.learning_rate)
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate))

    train_generator, validation_generator = get_generators(args, x_train, x_valid, y_train, y_valid)

    history = model.fit_generator(train_generator,
                                  validation_data=validation_generator,
                                  epochs=args.nb_epoch,
                                  callbacks=[checkpoint],
                                  verbose=1)
    # save the last model anyway (might not be the best)
    model.save(os.path.join(model_dir, "udacity_trained.h5"))

    dataset_folder = args.data_dir
    x_test = np.load(os.path.join(dataset_folder, 'udacity_weak_test_x.npy'))
    y_test = np.load(os.path.join(dataset_folder, 'udacity_weak_test_y.npy'))
    score = model.evaluate(x_test, y_test, verbose=0)
    return [score, score]


def s2b(s):
    """
    Converts a string to boolean value
    """
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'


def main(mutation_final_name):
    model_dir = os.path.join('trained_models')
    model_loc = os.path.join(model_dir, mutation_final_name)

    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory', dest='data_dir', type=str,
                        default=os.path.join('..', 'Datasets', 'Udacity'))
    parser.add_argument('-t', help='test size fraction', dest='test_size', type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability', dest='keep_prob', type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs', dest='nb_epoch', type=int, default=50)
    parser.add_argument('-s', help='samples per epoch', dest='samples_per_epoch', type=int, default=100)
    parser.add_argument('-b', help='batch size', dest='batch_size', type=int, default=64)
    parser.add_argument('-o', help='save best models only', dest='save_best_only', type=s2b, default='true')
    parser.add_argument('-l', help='learning rate', dest='learning_rate', type=float, default=1.0e-4)
    args = parser.parse_args()
    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    data = load_data(args)
    train_generator, validation_generator = get_generators(args, *data)

    if not os.path.exists(model_loc):
        print('model does not exist')
        import tensorflow as tf
        if tf.test.gpu_device_name():
            print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
        else:
            print("Please install GPU version of TF")
        """
        Load train/validation data set and train the model
        """

        model = build_model(args)
        score = train_model(model_dir, model, args, *data)
    else:
        print('model exists')
        model = tensorflow.keras.models.load_model(model_loc)
        dataset_folder = args.data_dir
        x_test = np.load(os.path.join(dataset_folder, 'udacity_weak_test_x.npy'))
        y_test = np.load(os.path.join(dataset_folder, 'udacity_weak_test_y.npy'))
        metric_value = model.evaluate(x_test, x_test, verbose=0)
        K.clear_session()
        score = [metric_value, metric_value]
    return score

if __name__ == '__main__':

    main('')
