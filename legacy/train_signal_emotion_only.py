import numpy as np
import os
import tensorflow as tf
from keras import Input, Model
from keras.layers import Conv1D, Flatten, LSTM, Dropout, \
    Dense
from keras.optimizers import SGD

from constants import EMOTIONS
from legacy.Framework import get_dataset, randomize_split, train, test

os.environ["CUDA_VISIBLE_DEVICES"] = ""

EPOCHS = 200

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
config = tf.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

data = get_dataset('signal-dataset.pkl')

training_data, testing_data = randomize_split(data)

x_train, y_emo_train = [], []

for d in training_data:
    x_train.append(d['x'])
    y_emo_train.append(d['emo'])

x_train = np.array(x_train)
y_emo_train = np.array(y_emo_train)

# make the model

# Shared Block
x = Input(shape=(44100, 1))

c1 = Conv1D(16, 2)(x)

lstm1 = LSTM(64, return_sequences=True)(c1)
dr1 = Dropout(0.3)(lstm1)
f2 = Flatten()(dr1)

emo_d2 = Dense(128, activation='relu')(f2)

d_out = Dense(len(EMOTIONS), activation='softmax', name='emotion_output')(emo_d2)

model = Model(inputs=x, outputs=[d_out], name='signal_CNN_LSTM')

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True),
              metrics=['accuracy'])

history, model = train(model, x_train.reshape((len(x_train), 44100, 1)), [y_emo_train], EPOCHS)

x_test, y_emo_test = [], []

for d in testing_data:
    x_test.append(d['x'])
    y_emo_test.append(d['emo'])

x_test = np.array(x_test)
y_emo_test = np.array(y_emo_test)

test(model, x_test.reshape((len(x_test), 44100, 1)), [y_emo_test])
