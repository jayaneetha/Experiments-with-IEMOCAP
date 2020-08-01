import librosa
import numpy as np
from enum import Enum

from Framework import get_dataset, randomize_split
from constants import NUM_MFCC, EMOTIONS


class FeatureType(Enum):
    MFCC = 1
    MEL = 2
    LOG_MEL = 3
    STFT = 4
    RAW = 5


def get_data(feature_type: FeatureType.MFCC):
    # data = get_dataset("signal-{}-class-dataset-4sec.pkl".format(len(EMOTIONS)))
    # data = get_dataset("signal-{}-class-dataset-4sec_sr_8k.pkl".format(len(EMOTIONS)))
    # data = get_dataset("signal-{}-class-dataset-8sec_sr_16k.pkl".format(len(EMOTIONS)))
    # data = get_dataset("signal-{}-class-dataset-2sec_sr_16k.pkl".format(len(EMOTIONS)))
    data = get_dataset("signal-no-silent-{}-class-dataset-2sec_sr_16k.pkl".format(len(EMOTIONS)))

    training_data, testing_data = randomize_split(data)
    x_train, y_emo_train, y_gen_train = [], [], []

    for d in training_data:
        feature = _get_feature(feature_type, d['x'])
        x_train.append(feature)
        y_emo_train.append(d['emo'])
        y_gen_train.append(d['gen'])

    x_train = np.array(x_train)
    y_emo_train = np.array(y_emo_train)
    y_gen_train = np.array(y_gen_train)

    x_test, y_emo_test, y_gen_test = [], [], []
    for d in testing_data:
        feature = _get_feature(feature_type, d['x'])
        x_test.append(feature)
        y_emo_test.append(d['emo'])
        y_gen_test.append(d['gen'])

    x_test = np.array(x_test)
    y_emo_test = np.array(y_emo_test)
    y_gen_test = np.array(y_gen_test)

    return (x_train, y_emo_train, y_gen_train), (x_test, y_emo_test, y_gen_test)


def _get_feature(feature_type: FeatureType, signal):
    if feature_type == FeatureType.MFCC:
        mfcc = librosa.feature.mfcc(signal, sr=22050, n_mfcc=NUM_MFCC)
        return mfcc

    if feature_type == FeatureType.MEL:
        melspectrogram = librosa.feature.melspectrogram(signal, sr=22050)
        return melspectrogram

    if feature_type == FeatureType.LOG_MEL:
        melspectrogram = librosa.feature.melspectrogram(signal, sr=22050)
        log_mel = np.log(melspectrogram)
        return log_mel

    if feature_type == FeatureType.STFT:
        stft = librosa.stft(signal)
        stft_db = librosa.amplitude_to_db(stft, ref=np.max)
        return stft_db

    if feature_type == FeatureType.RAW:
        return signal
