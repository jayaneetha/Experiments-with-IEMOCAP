import librosa
import numpy as np
import os
import tensorflow as tf
from keras import Input, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, \
    Dense, AveragePooling2D
from keras.optimizers import SGD

from constants import EMOTIONS, NUM_MFCC
from legacy.Framework import randomize_split, get_dataset, train, test

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

EPOCHS = 200

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
config = tf.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

data = get_dataset('signal-2-class-dataset.pkl')

# make the model

# Shared Block
x = Input(shape=(40, 87, 1))

c1 = Conv2D(3, kernel_size=8, padding='same', activation='relu')(x)
mp1 = MaxPooling2D(strides=2)(c1)

c2 = Conv2D(3, kernel_size=8, padding='same', activation='relu')(mp1)
mp2 = MaxPooling2D(strides=3)(c2)

c3 = Conv2D(3, kernel_size=8, padding='same', activation='relu')(mp2)
mp3 = MaxPooling2D(strides=4)(c3)

c2 = Conv2D(16, kernel_size=4, padding='same', activation='relu')(mp3)
ap1 = AveragePooling2D(pool_size=2)(c2)

f1 = Flatten()(ap1)

# emotion part
emo_d1 = Dense(512, activation='relu')(f1)
emo_dr1 = Dropout(0.3)(emo_d1)

d_out = Dense(len(EMOTIONS), activation='softmax', name='emotion_output')(emo_dr1)

model = Model(inputs=x, outputs=[d_out], name='mel_model_5')

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True),
              metrics=['accuracy'])

training_data, testing_data = randomize_split(data)

x_train, y_emo_train = [], []

for d in training_data:
    mfcc = librosa.feature.mfcc(d['x'], 22050, n_mfcc=NUM_MFCC)
    x_train.append(mfcc)
    y_emo_train.append(d['emo'])

x_train = np.array(x_train)
y_emo_train = np.array(y_emo_train)

history, model = train(model, x_train.reshape((len(x_train), 40, 87, 1)), [y_emo_train], EPOCHS)

x_test, y_emo_test = [], []

for d in testing_data:
    mfcc = librosa.feature.mfcc(d['x'], 22050, n_mfcc=NUM_MFCC)
    x_test.append(mfcc)
    y_emo_test.append(d['emo'])

x_test = np.array(x_test)
y_emo_test = np.array(y_emo_test)

test(model, x_test.reshape((len(x_test), 40, 87, 1)), [y_emo_test])
