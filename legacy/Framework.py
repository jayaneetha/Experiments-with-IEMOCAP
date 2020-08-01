import datetime
import pickle

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint


def train(model, x, y, EPOCHS, batch_size=4):
    print("Start Training")
    log_dir = "logs/fit/" + model.name + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    callback_list = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=1,
            mode='min'
        ),
        ModelCheckpoint(
            filepath=model.name + '.h5',
            monitor='val_acc',
            save_best_only='True',
            verbose=1,
            mode='max'
        ), tensorboard_callback]

    history = model.fit(x, y,
                        batch_size=batch_size, epochs=EPOCHS,
                        validation_split=0.2,
                        verbose=True,
                        callbacks=callback_list)
    return history, model


def test(model, x, y):
    matrices = model.evaluate(x, y)

    for i in range(len(model.metrics_names)):
        print("{} : \t {}".format(model.metrics_names[i], matrices[i]))


def randomize_split(data, split_ratio=0.8):
    # shuffle the dataset
    np.random.shuffle(data)

    # divide training and testing dataset
    training_count = int(len(data) * split_ratio)

    training_data = data[:training_count]
    testing_data = data[training_count:]
    return training_data, testing_data


def get_dataset(filename='dataset.pkl'):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        return data
