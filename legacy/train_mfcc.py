import numpy as np
import os
import tensorflow as tf
from keras import Input, Model
from keras.layers import TimeDistributed, BatchNormalization, Conv2D, MaxPooling2D, Flatten, LSTM, Dropout, \
    Dense, Bidirectional
from keras.optimizers import SGD

from constants import EMOTIONS, GENDERS
from legacy.Framework import randomize_split, train, test, get_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

EPOCHS = 1200

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
config = tf.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

data = get_dataset('dataset.pkl')

training_data, testing_data = randomize_split(data)

x_train, y_emo_train, y_gen_train = [], [], []

for d in training_data:
    x_train.append(d['x'])
    y_emo_train.append(d['emo'])
    y_gen_train.append(d['gen'])

x_train = np.array(x_train)
y_emo_train = np.array(y_emo_train)
y_gen_train = np.array(y_gen_train)

# make the model

# Shared Block
x = Input(shape=(40, 87, 1))

c1 = Conv2D(16, kernel_size=4, padding='same', activation='relu')(x)
mp1 = MaxPooling2D(pool_size=2)(c1)
bn1 = BatchNormalization()(mp1)

c2 = Conv2D(8, kernel_size=2, padding='same', activation='relu')(bn1)
mp2 = MaxPooling2D(pool_size=2)(c2)
bn2 = BatchNormalization()(mp2)

f1 = TimeDistributed(Flatten())(bn2)
bi_lstm = Bidirectional(LSTM(256, return_sequences=True))(f1)
lstm = LSTM(50, return_sequences=True)(bi_lstm)
dr1 = Dropout(0.3)(lstm)
f2 = Flatten()(dr1)

# emotion part
emo_d1 = Dense(512, activation='relu')(f2)
emo_d2 = Dense(256, activation='relu')(emo_d1)
emo_dr1 = Dropout(0.3)(emo_d2)
emo_d2 = Dense(128, activation='relu')(emo_dr1)
emo_d3 = Dense(len(EMOTIONS), activation='softmax', name='emotion_output')(emo_d2)

# gender part
gen_d1 = Dense(256, activation='relu')(f2)
gen_dr1 = Dropout(0.3)(gen_d1)
gen_d2 = Dense(128, activation='relu')(gen_dr1)
gen_d3 = Dense(len(GENDERS), activation='softmax', name='gender_output')(gen_d2)

model = Model(inputs=x, outputs=[emo_d3, gen_d3], name='IEMOCAP_multitask')

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True),
              metrics=['accuracy'])

history, model = train(model, x_train.reshape((len(x_train), 40, 87, 1)), [y_emo_train, y_gen_train], EPOCHS)

x_test, y_emo_test, y_gen_test = [], [], []

for d in testing_data:
    x_test.append(d['x'])
    y_emo_test.append(d['emo'])
    y_gen_test.append(d['gen'])

x_test = np.array(x_test)
y_emo_test = np.array(y_emo_test)
y_gen_test = np.array(y_gen_test)

test(model, x_test.reshape((len(x_test), 40, 87, 1)), [y_emo_test, y_gen_test])
