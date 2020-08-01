import librosa
import numpy as np
import os
import tensorflow as tf
from keras import Input, Model
from keras.layers import TimeDistributed, BatchNormalization, Conv2D, MaxPooling2D, Flatten, LSTM, Dropout, \
    Dense, Bidirectional, concatenate
from keras.optimizers import SGD

from constants import EMOTIONS
from legacy.Framework import train, randomize_split, test, get_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

EPOCHS = 200

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
config = tf.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

data = get_dataset('signal-dataset.pkl')

training_data, testing_data = randomize_split(data)

x_train_mel, x_train_mfcc, y_emo_train = [], [], []

for d in training_data:
    melspectrogram = librosa.feature.melspectrogram(d['x'], 22050)
    log_mel = np.log(melspectrogram)
    x_train_mel.append(log_mel)

    mfcc = librosa.feature.mfcc(d['x'], 22050, n_mfcc=128)
    x_train_mfcc.append(mfcc)

    y_emo_train.append(d['emo'])

x_train_mel = np.array(x_train_mel)
x_train_mfcc = np.array(x_train_mfcc)

y_emo_train = np.array(y_emo_train)

# make the model

# Shared Block
x_mel = Input(shape=(128, 87, 1))
x_mfcc = Input(shape=(128, 87, 1))

x = concatenate([x_mel, x_mfcc])

c1 = Conv2D(32, kernel_size=8, padding='same', activation='relu')(x_mfcc)
bn1 = BatchNormalization()(c1)

c3 = Conv2D(8, kernel_size=2, padding='same', activation='relu')(bn1)

c4 = Conv2D(8, kernel_size=1, padding='same', activation='relu')(c3)
mp4 = MaxPooling2D(pool_size=2)(c4)

f1 = TimeDistributed(Flatten())(mp4)
bi_lstm = Bidirectional(LSTM(16, return_sequences=True))(f1)

dr1 = Dropout(0.3)(bi_lstm)
f2 = Flatten()(dr1)

# emotion part
emo_d1 = Dense(1024, activation='relu')(f2)
emo_d2 = Dense(256, activation='tanh')(emo_d1)
emo_d3 = Dense(128, activation='tanh')(emo_d2)
emo_dr1 = Dropout(0.3)(emo_d3)
emo_d5 = Dense(128, activation='relu')(emo_dr1)

d_out = Dense(len(EMOTIONS), activation='softmax', name='emotion_output')(emo_d5)

model = Model(inputs=[x_mfcc, x_mel], outputs=[d_out], name='mfcc_mel_model4')

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True),
              metrics=['accuracy'])

history, model = train(model, [x_train_mel.reshape((len(x_train_mel), 128, 87, 1)),
                               x_train_mfcc.reshape((len(x_train_mfcc), 128, 87, 1))], [y_emo_train], EPOCHS)

x_test_mel, x_test_mfcc, y_emo_test = [], [], []

for d in testing_data:
    melspectrogram = librosa.feature.melspectrogram(d['x'], 22050)
    x_test_mel.append(melspectrogram)

    mfcc = librosa.feature.mfcc(d['x'], 22050, n_mfcc=128)
    x_test_mfcc.append(mfcc)

    y_emo_test.append(d['emo'])

x_test_mel = np.array(x_test_mel)
x_test_mfcc = np.array(x_test_mfcc)
y_emo_test = np.array(y_emo_test)

test(model, [x_test_mel.reshape((len(x_test_mel), 128, 87, 1)), x_test_mfcc.reshape((len(x_test_mfcc), 128, 87, 1))],
     [y_emo_test])
