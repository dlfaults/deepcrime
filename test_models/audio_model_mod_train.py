import os
import shutil
import numpy as np

import tensorflow as tf
from tensorflow import keras

from pathlib import Path

SAMPLING_RATE = 16000


def load_noise_sample(path):
    sample, sampling_rate = tf.audio.decode_wav(
        tf.io.read_file(path), desired_channels=1
    )
    if sampling_rate == SAMPLING_RATE:
        # Number of slices of 16000 each that can be generated from the noise sample
        slices = int(sample.shape[0] / SAMPLING_RATE)
        sample = tf.split(sample[: slices * SAMPLING_RATE], slices)
        return sample
    else:
        print("Sampling rate for {} is incorrect. Ignoring it".format(path))
        return None


def paths_and_labels_to_dataset(audio_paths, labels):
    """Constructs a dataset of audios and labels."""
    path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
    audio_ds = path_ds.map(lambda x: path_to_audio(x))
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    return tf.data.Dataset.zip((audio_ds, label_ds))


def path_to_audio(path):
    """Reads and decodes an audio file."""
    audio = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio, 1, SAMPLING_RATE)
    return audio


def add_noise(audio, noises=None, scale=0.5):
    if noises is not None:
        # Create a random tensor of the same size as audio ranging from
        # 0 to the number of noise stream samples that we have.
        tf_rnd = tf.random.uniform(
            (tf.shape(audio)[0],), 0, noises.shape[0], dtype=tf.int32
        )
        noise = tf.gather(noises, tf_rnd, axis=0)

        # Get the amplitude proportion between the audio and the noise
        prop = tf.math.reduce_max(audio, axis=1) / tf.math.reduce_max(noise, axis=1)
        prop = tf.repeat(tf.expand_dims(prop, axis=1), tf.shape(audio)[1], axis=1)

        print('audio shape:', audio.shape)
        print('noise shape:', noise.shape)
        print((noise * prop * scale).shape)
        # Adding the rescaled noise to audio
        audio = audio + noise * prop * scale

    return audio


def audio_to_fft(audio):
    # Since tf.signal.fft applies FFT on the innermost dimension,
    # we need to squeeze the dimensions and then expand them again
    # after FFT
    audio = tf.squeeze(audio, axis=-1)
    fft = tf.signal.fft(
        tf.cast(tf.complex(real=audio, imag=tf.zeros_like(audio)), tf.complex64)
    )
    fft = tf.expand_dims(fft, axis=-1)

    # Return the absolute value of the first half of the FFT
    # which represents the positive frequencies
    return tf.math.abs(fft[:, : (audio.shape[1] // 2), :])


def residual_block(x, filters, conv_num=3, activation="relu"):
    # Shortcut
    s = keras.layers.Conv1D(filters, 1, padding="same")(x)
    for i in range(conv_num - 1):
        x = keras.layers.Conv1D(filters, 3, padding="same")(x)
        x = keras.layers.Activation(activation)(x)
    x = keras.layers.Conv1D(filters, 3, padding="same")(x)
    x = keras.layers.Add()([x, s])
    x = keras.layers.Activation(activation)(x)
    return keras.layers.MaxPool1D(pool_size=2, strides=2)(x)


def build_model(input_shape, num_classes):
    inputs = keras.layers.Input(shape=input_shape, name="input")

    x = residual_block(inputs, 16, 2)
    x = residual_block(x, 32, 2)
    x = residual_block(x, 64, 3)
    x = residual_block(x, 128, 3)
    x = residual_block(x, 128, 3)

    x = keras.layers.AveragePooling1D(pool_size=3, strides=3)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dense(128, activation="relu")(x)

    outputs = keras.layers.Dense(num_classes, activation="softmax", name="output")(x)

    return keras.models.Model(inputs=inputs, outputs=outputs)


def get_all_data():
    print(os.getcwd())
    DATASET_ROOT = os.path.join('..', 'Datasets', 'Audio', '16000_pcm_speeches')

    BATCH_SIZE = 128
    # The folders in which we will put the audio samples and the noise samples
    AUDIO_SUBFOLDER = "audio"
    NOISE_SUBFOLDER = "noise"

    DATASET_AUDIO_PATH = os.path.join(DATASET_ROOT, AUDIO_SUBFOLDER)
    DATASET_NOISE_PATH = os.path.join(DATASET_ROOT, NOISE_SUBFOLDER)

    # Percentage of samples to use for validation
    VALID_SPLIT = 0.1

    # Percentage of samples to use for testing
    TEST_SPLIT = 0.2

    # Seed to use when shuffling the dataset and the noise
    SHUFFLE_SEED = 43

    # The sampling rate to use.
    # This is the one used in all of the audio samples.
    # We will resample all of the noise to this sampling rate.
    # This will also be the output size of the audio wave samples
    # (since all samples are of 1 second long)
    # The factor to multiply the noise with according to:
    #   noisy_sample = sample + noise * prop * scale
    #      where prop = sample_amplitude / noise_amplitude
    SCALE = 0.5


    # If folder `audio`, does not exist, create it, otherwise do nothing
    if os.path.exists(DATASET_AUDIO_PATH) is False:
        os.makedirs(DATASET_AUDIO_PATH)

    # If folder `noise`, does not exist, create it, otherwise do nothing
    if os.path.exists(DATASET_NOISE_PATH) is False:
        os.makedirs(DATASET_NOISE_PATH)

    for folder in os.listdir(DATASET_ROOT):
        if os.path.isdir(os.path.join(DATASET_ROOT, folder)):
            if folder in [AUDIO_SUBFOLDER, NOISE_SUBFOLDER]:
                # If folder is `audio` or `noise`, do nothing
                continue
            elif folder in ["other", "_background_noise_"]:
                # If folder is one of the folders that contains noise samples,
                # move it to the `noise` folder
                shutil.move(
                    os.path.join(DATASET_ROOT, folder),
                    os.path.join(DATASET_NOISE_PATH, folder),
                )
            else:
                # Otherwise, it should be a speaker folder, then move it to
                # `audio` folder
                shutil.move(
                    os.path.join(DATASET_ROOT, folder),
                    os.path.join(DATASET_AUDIO_PATH, folder),
                )

    # Get the list of all noise files
    noise_paths = []
    for subdir in os.listdir(DATASET_NOISE_PATH):
        subdir_path = Path(DATASET_NOISE_PATH) / subdir
        if os.path.isdir(subdir_path):
            noise_paths += [
                os.path.join(subdir_path, filepath)
                for filepath in os.listdir(subdir_path)
                if filepath.endswith(".wav")
            ]

    print(
        "Found {} files belonging to {} directories".format(
            len(noise_paths), len(os.listdir(DATASET_NOISE_PATH))
        )
    )

    command = (
            "for dir in `ls -1 " + DATASET_NOISE_PATH + "`; do "
                                                        "for file in `ls -1 " + DATASET_NOISE_PATH + "/$dir/*.wav`; do "
                                                                                                     "sample_rate=`ffprobe -hide_banner -loglevel panic -show_streams "
                                                                                                     "$file | grep sample_rate | cut -f2 -d=`; "
                                                                                                     "if [ $sample_rate -ne 16000 ]; then "
                                                                                                     "ffmpeg -hide_banner -loglevel panic -y "
                                                                                                     "-i $file -ar 16000 temp.wav; "
                                                                                                     "mv temp.wav $file; "
                                                                                                     "fi; done; done"
    )

    os.system(command)

    noises = []
    for path in noise_paths:
        sample = load_noise_sample(path)
        if sample:
            noises.extend(sample)
    noises = tf.stack(noises)

    print(
        "{} noise files were split into {} noise samples where each is {} sec. long".format(
            len(noise_paths), noises.shape[0], noises.shape[1] // SAMPLING_RATE
        )
    )

    # Get the list of audio file paths along with their corresponding labels
    # Get the list of audio file paths along with their corresponding labels

    #class_names = os.listdir(DATASET_AUDIO_PATH)
    class_names = ['Julia_Gillard', 'Nelson_Mandela', 'Benjamin_Netanyau', 'Magaret_Tarcher', 'Jens_Stoltenberg']
    print("Our class names: {}".format(class_names, ))

    audio_paths = []
    labels = []
    for label, name in enumerate(class_names):
        print("Processing speaker {}".format(name, ))
        dir_path = Path(DATASET_AUDIO_PATH) / name
        speaker_sample_paths = [
            os.path.join(dir_path, filepath)
            for filepath in os.listdir(dir_path)
            if filepath.endswith(".wav")
        ]
        audio_paths += speaker_sample_paths
        labels += [label] * len(speaker_sample_paths)

    print(
        "Found {} files belonging to {} classes.".format(len(audio_paths), len(class_names))
    )

    # Shuffle
    rng = np.random.RandomState(SHUFFLE_SEED)
    rng.shuffle(audio_paths)
    rng = np.random.RandomState(SHUFFLE_SEED)
    rng.shuffle(labels)

    # Split into training and validation
    num_val_samples = int(VALID_SPLIT * len(audio_paths))
    num_test_samples = int(TEST_SPLIT * (len(audio_paths) - num_val_samples))
    print("Using {} files for training.".format(len(audio_paths) - num_val_samples))

    train_audio_paths = audio_paths[:-num_val_samples]
    train_labels = labels[:-num_val_samples]
    #print(train_audio_paths[0:5])

    print("Using {} files for validation.".format(num_val_samples))
    valid_audio_paths = audio_paths[-num_val_samples:]
    valid_labels = labels[-num_val_samples:]

    test_audio_paths = train_audio_paths[-num_test_samples:]
    test_labels = train_labels[-num_test_samples:]

    train_audio_paths = train_audio_paths[:-num_test_samples]
    train_labels = train_labels[:-num_test_samples]

    train_audio_paths: 'x_train'
    train_labels: 'y_train'


    # Create 2 datasets, one for training and the other for validation
    train_ds = paths_and_labels_to_dataset(train_audio_paths, train_labels)
    train_ds = train_ds.shuffle(buffer_size=BATCH_SIZE * 8, seed=SHUFFLE_SEED).batch(
        BATCH_SIZE
    )

    valid_ds = paths_and_labels_to_dataset(valid_audio_paths, valid_labels)
    valid_ds = valid_ds.shuffle(buffer_size=32 * 8, seed=SHUFFLE_SEED).batch(32)

    test_ds = paths_and_labels_to_dataset(test_audio_paths, test_labels)
    test_ds = test_ds.shuffle(buffer_size=32 * 8, seed=SHUFFLE_SEED).batch(32)

    # Add noise to the training set
    train_ds = train_ds.map(
        lambda x, y: (add_noise(x, noises, scale=SCALE), y),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    # Transform audio wave to the frequency domain using `audio_to_fft`
    train_ds = train_ds.map(
        lambda x, y: (audio_to_fft(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

    valid_ds = valid_ds.map(
        lambda x, y: (audio_to_fft(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    valid_ds = valid_ds.prefetch(tf.data.experimental.AUTOTUNE)

    test_ds = test_ds.map(
        lambda x, y: (audio_to_fft(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)

    return train_ds, test_ds, valid_ds, class_names

def main(model_name):
    BATCH_SIZE = 128
    EPOCHS = 100
    #model_dir = "/home/ubuntu/mutation-tool/trained_models/"
    model_dir = os.path.join('trained_models')
    model_location = os.path.join(model_dir, model_name)

    train_ds, test_ds, valid_ds, class_names = get_all_data()

    weak_ts_loc = os.path.join('..', 'Datasets', 'Audio')
    weak_ts_x = np.load(os.path.join(weak_ts_loc, 'audio_easy_x.npy'))
    weak_ts_y = np.load(os.path.join(weak_ts_loc, 'audio_easy_y.npy'))
    weak_test_ds = tf.data.Dataset.zip(
        (tf.data.Dataset.from_tensor_slices(weak_ts_x), tf.data.Dataset.from_tensor_slices(weak_ts_y)))
    weak_test_ds = weak_test_ds.batch(32)
    weak_test_ds = weak_test_ds.prefetch(tf.data.experimental.AUTOTUNE)

    if not os.path.exists(model_location):
        model = build_model((SAMPLING_RATE // 2, 1), len(class_names))
        model.summary()
        # Compile the model using Adam's default learning rate
        model.compile(
            optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )
        # Add callbacks:
        # 'EarlyStopping' to stop training when the model is not enhancing anymore
        # 'ModelCheckPoint' to always keep the model that has the best val_accuracy
        model_save_filename = "audio_trained.h5"

        earlystopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        mdlcheckpoint_cb = keras.callbacks.ModelCheckpoint(
            model_save_filename, monitor="val_accuracy", save_best_only=True
        )

        history = model.fit(
            train_ds,
            epochs=EPOCHS,
            validation_data=valid_ds,
            callbacks=[earlystopping_cb, mdlcheckpoint_cb],
        )

        model.save(model_location)
        score = model.evaluate(weak_test_ds, verbose=0)
    else:
        model = tf.keras.models.load_model(model_location, compile=True)
        score = model.evaluate(weak_test_ds, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return score

if __name__ == "__main__":
    main('audio_original_0.h5')
