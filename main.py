import datetime
import socket

import numpy as np
import os
import tensorflow as tf
from keras import Input, Model
from keras.optimizers import RMSprop

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
DRAW_CONFUSION_MATRIX = False
SAVE_MODELS = True

TIME = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def train_mfcc():
    TIME = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    input_layer = Input(shape=(NUM_MFCC, NO_features, 1))
    model = _get_model(input_layer, 'mfcc')

    (x_train, y_emo_train, y_gen_train), (x_test, y_emo_test, y_gen_test) = get_data(feature_types=[FeatureType.MFCC])

    train_mfccs = np.array([d[FeatureType.MFCC.name] for d in x_train])
    train_mfccs = train_mfccs.reshape((len(train_mfccs), NUM_MFCC, NO_features, 1))

    history, model, training_time = train(model, [train_mfccs], [y_emo_train, y_gen_train], EPOCHS, batch_size=16,
                                          TIME=TIME)

    test_mfccs = np.array([d[FeatureType.MFCC.name] for d in x_test])
    test_mfccs = test_mfccs.reshape((len(test_mfccs), NUM_MFCC, NO_features, 1))

    test_results = test(model, [test_mfccs], [y_emo_test, y_gen_test])

    if DRAW_CONFUSION_MATRIX:
        _draw_confusion_matrix(model, [test_mfccs], y_emo_test, 'mfcc', EMOTIONS, 'Emotions', 0, time=TIME)
        _draw_confusion_matrix(model, [test_mfccs], y_gen_test, 'mfcc', GENDERS, 'Gender', 1, time=TIME)

    if SAVE_MODELS:
        _save_model(model, 'mfcc', TIME)

    return test_results, training_time, model


def train_mfcc_single():
    input_layer = Input(shape=(NUM_MFCC, NO_features, 1))
    model = _get_model(input_layer, 'mfcc')

    (x_train, y_emo_train, y_gen_train), (x_test, y_emo_test, y_gen_test) = get_data(feature_types=[FeatureType.MFCC])

    history, model = train(model, x_train.reshape((len(x_train), NUM_MFCC, NO_features, 1)), [y_emo_train],
                           EPOCHS,
                           batch_size=16)

    test_results = test(model, x_test.reshape((len(x_test), NUM_MFCC, NO_features, 1)), [y_emo_test])
    _draw_confusion_matrix(model, x_test, y_emo_test, 'mfcc', EMOTIONS, 'Emotions')

    if SAVE_MODELS:
        _save_model(model, 'mfcc')
    return test_results


def train_mel(mel_type='mel'):
    input_layer = Input(shape=(128, 87, 1))
    model = _get_model(input_layer, mel_type)

    (x_train, y_emo_train, y_gen_train), (x_test, y_emo_test, y_gen_test) = get_data(feature_types=[FeatureType.MEL])

    history, model = train(model, x_train.reshape((len(x_train), 128, 87, 1)), [y_emo_train], EPOCHS)

    test_results = test(model, x_test.reshape((len(x_test), 128, 87, 1)), [y_emo_test])
    _draw_confusion_matrix(model, x_test, y_emo_test, mel_type, EMOTIONS, 'Emotions')
    return test_results


def train_stft():
    input_layer = Input(shape=(1025, 87, 1))
    model = _get_model(input_layer, 'stft')

    (x_train, y_emo_train, y_gen_train), (x_test, y_emo_test, y_gen_test) = get_data(feature_types=[FeatureType.STFT])

    history, model = train(model, x_train.reshape((len(x_train), 1025, 87, 1)), [y_emo_train], EPOCHS)

    test_results = test(model, x_test.reshape((len(x_test), 1025, 87, 1)), [y_emo_test])
    _draw_confusion_matrix(model, x_test, y_emo_test, 'stft', EMOTIONS, 'Emotions')
    return test_results


def _draw_confusion_matrix(model: Model, x_test, y_test, feature_type, class_list, class_type, prediction_index=None,
                           time=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")):
    cm = get_confusion_matrix(model, x_test, y_test, prediction_index)
    plot_confusion_matrix(cm, class_list,
                          "img/CM/Week_{}/CM_{}_classes_{}_{}_{}_{}.png".format(WEEK, model.name, len(class_list),
                                                                                class_type,
                                                                                type(model.optimizer).__name__,
                                                                                time),
                          title="Confusion Matrix Input Features: {}, Model Name :{} Classes: {}".format(
                              str(feature_type),
                              model.name, class_type))


def _save_model(model: Model, feature_type, time=TIME):
    base_dir = "models/Week_{}/".format(WEEK)
    model_name = "{}_{}_classes_{}_{}_{}".format(model.name, feature_type, len(EMOTIONS),
                                                 type(model.optimizer).__name__,
                                                 time)
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
    model = models.get_model_9_multi(input_layer, model_name_prefix=mel_type)

    model.compile(loss='categorical_crossentropy',
                  # optimizer=SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True),
                  # optimizer=Adam(),
                  optimizer=RMSprop(),
                  # optimizer=Adagrad(),
                  metrics=['accuracy'])

    if host != 'asimov':
        plot_model(model, model.name)
    model.reset_states()
    model.summary(line_length=100)
    return model


def group_eval(func, name):
    train_episodes = 5
    test_results_collection = []
    total_training_times = []
    model = None

    for i in range(train_episodes):
        test_result, training_time, model = func()
        test_results_collection.append(test_result)
        total_training_times.append(training_time)

    import json
    filename = "group_eval/week_{}/{}.json".format(str(WEEK), name)
    print("Saving group eval : {}".format(filename))
    with open(filename, 'w') as f:
        json.dump(test_results_collection, f)

    total_training_times = np.array(total_training_times)

    print("Mean Training Time: {} seconds".format(str(total_training_times.mean())))

    (x_train, y_emo_train, y_gen_train), (x_test, y_emo_test, y_gen_test) = get_data(feature_types=[FeatureType.MFCC])

    test_mfccs = np.array([d[FeatureType.MFCC.name] for d in x_test])
    test_mfccs = test_mfccs.reshape((len(test_mfccs), NUM_MFCC, NO_features, 1))

    _draw_confusion_matrix(model, [test_mfccs], y_emo_test, 'mfcc', EMOTIONS, 'Emotions', 0, time=TIME)
    _draw_confusion_matrix(model, [test_mfccs], y_gen_test, 'mfcc', GENDERS, 'Gender', 1, time=TIME)

    return test_results_collection


def print_final_scores(output_name, acc_list: np.ndarray):
    print("{} : ******".format(output_name))
    print("\t mean : {}".format(acc_list.mean()))
    print("\t std.dev : {}".format(acc_list.std()))
    print("\t var : {}".format(acc_list.var()))


if __name__ == "__main__":
    print("Start")
    test_results_collection = group_eval(train_mfcc, 'model_9_multi_rmsprop_' + TIME)
    # train_mfcc()
    # train_mel()
    # train_mel('log-mel')
    # train_stft()

    emo_acc = np.array([d['emotion_output_accuracy'] for d in test_results_collection])
    gen_acc = np.array([d['gender_output_accuracy'] for d in test_results_collection])
    print_final_scores('emotion', emo_acc)
    print_final_scores('gender', gen_acc)
