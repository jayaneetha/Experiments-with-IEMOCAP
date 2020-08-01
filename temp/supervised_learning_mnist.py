# import imageio
import datetime
import glob

import numpy as np
import os
import random
import tensorflow as tf
from PIL import Image
from keras import Input, Model
from keras.layers import Conv2D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import SGD
from keras.utils import to_categorical

COLORS = {
    'red': 0,
    'green': 1,
    'blue': 2
}

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

EPOCHS = 1200

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
config = tf.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)


def get_dataset(file_paths):
    X = []
    y_color = []
    y_label = []

    for path in file_paths:
        clr = path.split('/')[-1].split('_')[0]
        lab = path.split('/')[-2]

        im_frame = Image.open(path)
        np_frame = np.array(im_frame.getdata())
        im = np_frame.reshape((28, 28, 3))

        X.append(im)
        y_color.append(COLORS[clr])
        y_label.append(int(lab))

    X = np.array(X)
    y_color = to_categorical(y_color, num_classes=3)
    y_label = to_categorical(y_label, num_classes=10)

    return X, y_label, y_color


training_paths = glob.glob("/home/u1116888/projects/iemocap_dataset/temp/colorized-MNIST/training/*/*.png")
random.shuffle(training_paths)

testing_paths = glob.glob("/home/u1116888/projects/iemocap_dataset/temp/colorized-MNIST/testing/*/*.png")
random.shuffle(testing_paths)

training_x, training_y_label, training_y_color = get_dataset(training_paths)

# Model Block

# Shared Block
x = Input(shape=(28, 28, 3))

c1 = Conv2D(32, kernel_size=3, activation='relu')(x)
bn1 = BatchNormalization()(c1)
c2 = Conv2D(32, kernel_size=3, activation='relu')(bn1)
bn2 = BatchNormalization()(c2)
c3 = Conv2D(32, kernel_size=5, strides=2, padding='same', activation='relu')(bn2)
bn3 = BatchNormalization()(c3)
do1 = Dropout(0.4)(bn3)

c4 = Conv2D(64, kernel_size=3, activation='relu')(do1)
bn4 = BatchNormalization()(c4)
c5 = Conv2D(64, kernel_size=3, activation='relu')(bn4)
bn5 = BatchNormalization()(c5)
c6 = Conv2D(64, kernel_size=5, strides=2, padding='same', activation='relu')(bn5)
bn6 = BatchNormalization()(c6)
do2 = Dropout(0.4)(bn6)
f = Flatten()(do2)

# digit recognizer
dr_d1 = Dense(512, activation='relu')(f)
dr_bn1 = BatchNormalization()(dr_d1)
dr_d2 = Dense(256, activation='relu')(dr_bn1)
dr_do1 = Dropout(0.3)(dr_d2)
dr_d3 = Dense(10, activation='softmax', name='recognizer_output')(dr_do1)

# background color recognizer
bgr_d1 = Dense(256, activation='relu')(f)
bgr_d2 = Dense(3, activation='softmax', name='background_color_output')(bgr_d1)

model = Model(inputs=x, outputs=[dr_d3, bgr_d2], name='MNIST_multitask')

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True),
              metrics=['accuracy', 'mse'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(training_x.reshape((len(training_x), 28, 28, 3)), [training_y_label, training_y_color],
                    batch_size=64, epochs=EPOCHS,
                    validation_split=0.2,
                    verbose=True,
                    callbacks=[tensorboard_callback])

testing_x, testing_y_label, testing_y_color = get_dataset(testing_paths)

matrices = model.evaluate(testing_x.reshape((len(testing_x), 28, 28, 3)), [testing_y_label, testing_y_color])

for i in range(len(model.metrics_names)):
    print("{} : \t {}".format(model.metrics_names[i], matrices[i]))
