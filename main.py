import datetime
import socket

import os
import tensorflow as tf
from keras import Input, Model
from keras.optimizers import Adam

import models
from Framework import train, test, get_confusion_matrix, plot_confusion_matrix, plot_model
from constants import EMOTIONS, NUM_MFCC, NO_features, WEEK, GENDERS
from data import get_data, FeatureType

host = socket.gethostname()

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
config = tf.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

EPOCHS = 200
DRAW_CONFUSION_MATRIX = True
SAVE_MODELS = True


def train_mfcc():
    input_layer = Input(shape=(NUM_MFCC, NO_features, 1))
    model = _get_model(input_layer, 'mfcc')

    (x_train, y_emo_train, y_gen_train), (x_test, y_emo_test, y_gen_test) = get_data(feature_type=FeatureType.MFCC)

    history, model = train(model, x_train.reshape((len(x_train), NUM_MFCC, NO_features, 1)), [y_emo_train, y_gen_train],
                           EPOCHS,
                           batch_size=16)

    test(model, x_test.reshape((len(x_test), NUM_MFCC, NO_features, 1)), [y_emo_test, y_gen_test])
    _draw_confusion_matrix(model, x_test, y_emo_test, 'mfcc', EMOTIONS, 'Emotions', 0)
    _draw_confusion_matrix(model, x_test, y_gen_test, 'mfcc', GENDERS, 'Gender', 1)

    if SAVE_MODELS:
        _save_model(model, 'mfcc')


def train_mfcc_single():
    input_layer = Input(shape=(NUM_MFCC, NO_features, 1))
    model = _get_model(input_layer, 'mfcc')

    (x_train, y_emo_train, y_gen_train), (x_test, y_emo_test, y_gen_test) = get_data(feature_type=FeatureType.MFCC)

    history, model = train(model, x_train.reshape((len(x_train), NUM_MFCC, NO_features, 1)), [y_emo_train],
                           EPOCHS,
                           batch_size=16)

    test(model, x_test.reshape((len(x_test), NUM_MFCC, NO_features, 1)), [y_emo_test])
    _draw_confusion_matrix(model, x_test, y_emo_test, 'mfcc', EMOTIONS, 'Emotions')

    if SAVE_MODELS:
        _save_model(model, 'mfcc')


def train_mel(mel_type='mel'):
    input_layer = Input(shape=(128, 87, 1))
    model = _get_model(input_layer, mel_type)

    (x_train, y_emo_train, y_gen_train), (x_test, y_emo_test, y_gen_test) = get_data(feature_type=FeatureType.MEL)

    history, model = train(model, x_train.reshape((len(x_train), 128, 87, 1)), [y_emo_train], EPOCHS)

    test(model, x_test.reshape((len(x_test), 128, 87, 1)), [y_emo_test])
    _draw_confusion_matrix(model, x_test, y_emo_test, mel_type, EMOTIONS, 'Emotions')


def train_stft():
    input_layer = Input(shape=(1025, 87, 1))
    model = _get_model(input_layer, 'stft')

    (x_train, y_emo_train, y_gen_train), (x_test, y_emo_test, y_gen_test) = get_data(feature_type=FeatureType.STFT)

    history, model = train(model, x_train.reshape((len(x_train), 1025, 87, 1)), [y_emo_train], EPOCHS)

    test(model, x_test.reshape((len(x_test), 1025, 87, 1)), [y_emo_test])
    _draw_confusion_matrix(model, x_test, y_emo_test, 'stft', EMOTIONS, 'Emotions')


def _draw_confusion_matrix(model: Model, x_test, y_test, feature_type, class_list, class_type, prediction_index=None):
    cm = get_confusion_matrix(model, x_test, y_test, prediction_index)
    plot_confusion_matrix(cm, class_list,
                          "img/CM/Week_{}/CM_{}_classes_{}_{}_{}_{}.png".format(WEEK, model.name, len(class_list),
                                                                                class_type,
                                                                                type(model.optimizer).__name__,
                                                                                datetime.datetime.now().strftime(
                                                                                    "%Y%m%d-%H%M%S")),
                          title="Confusion Matrix Input Features: {}, Model Name :{} Classes: {}".format(
                              str(feature_type),
                              model.name, class_type))


def _save_model(model: Model, feature_type):
    base_dir = "models/Week_{}/".format(WEEK)
    model_name = "{}_{}_classes_{}_{}_{}".format(model.name, feature_type, len(EMOTIONS),
                                                 type(model.optimizer).__name__,
                                                 datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    path = "{}/{}".format(base_dir, model_name)
    os.mkdir(path)
    # serialize model to JSON
    model_json = model.to_json()
    with open(path + "/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(path + "/model.h5")
    print("Saved model to disk: {}".format(path))


def _get_model(input_layer, mel_type):
    model = models.get_model_14_multi(input_layer, model_name_prefix=mel_type)

    model.compile(loss='categorical_crossentropy',
                  # optimizer=SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True),
                  optimizer=Adam(),
                  # optimizer=RMSprop(),
                  # optimizer=Adagrad(),
                  metrics=['accuracy'])

    if host != 'asimov':
        plot_model(model, model.name)

    model.summary(line_length=100)
    return model


if __name__ == "__main__":
    print("Start")
    train_mfcc()
    # train_mel()
    # train_mel('log-mel')
    # train_stft()
