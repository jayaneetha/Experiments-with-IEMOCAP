import librosa
import numpy as np
from enum import Enum

from Framework import get_dataset, randomize_split
from constants import NUM_MFCC, EMOTIONS, SR, DURATION
from legacy.convert_iemocap_dataset_to_pkl import split_audio


class FeatureType(Enum):
    MFCC = 1
    MEL = 2
    LOG_MEL = 3
    STFT = 4
    RAW = 5
    PITCH = 6


def get_data(feature_types: [FeatureType.MFCC], data=None, process_test_set=False):
    # data = get_dataset("signal-{}-class-dataset-4sec.pkl".format(len(EMOTIONS)))
    # data = get_dataset("signal-{}-class-dataset-4sec_sr_8k.pkl".format(len(EMOTIONS)))
    # data = get_dataset("signal-{}-class-dataset-8sec_sr_16k.pkl".format(len(EMOTIONS)))
    # data = get_dataset("signal-{}-class-dataset-2sec_sr_16k.pkl".format(len(EMOTIONS)))
    # data = get_dataset("signal-no-silent-{}-class-dataset-2sec_sr_16k.pkl".format(len(EMOTIONS)))
    if data is None:
        data = get_dataset("signal-no-silent-{}-class-dataset-2sec_sr_22k.pkl".format(len(EMOTIONS)))

    training_data, testing_data = randomize_split(data)
    x_train, y_emo_train, y_gen_train = [], [], []

    for d in training_data:
        training_features = {}

        for feature_type in feature_types:
            feature = _get_feature(feature_type, d['x'])
            training_features[feature_type.name] = feature

        x_train.append(training_features)
        y_emo_train.append(d['emo'])
        y_gen_train.append(d['gen'])

    x_train = np.array(x_train)
    y_emo_train = np.array(y_emo_train)
    y_gen_train = np.array(y_gen_train)

    x_test, y_emo_test, y_gen_test = [], [], []

    if process_test_set:
        for d in testing_data:
            testing_features = {}

            for feature_type in feature_types:
                feature = _get_feature(feature_type, d['x'])
                testing_features[feature_type.name] = feature

            x_test.append(testing_features)
            y_emo_test.append(d['emo'])
            y_gen_test.append(d['gen'])

        x_test = np.array(x_test)
        y_emo_test = np.array(y_emo_test)
        y_gen_test = np.array(y_gen_test)

    return (x_train, y_emo_train, y_gen_train), (x_test, y_emo_test, y_gen_test)


def _get_feature(feature_type: FeatureType, signal):
    if feature_type == FeatureType.MFCC:
        mfcc = librosa.feature.mfcc(signal, sr=SR, n_mfcc=NUM_MFCC)
        return mfcc

    if feature_type == FeatureType.MEL:
        melspectrogram = librosa.feature.melspectrogram(signal, sr=SR)
        return melspectrogram

    if feature_type == FeatureType.LOG_MEL:
        melspectrogram = librosa.feature.melspectrogram(signal, sr=SR)
        log_mel = np.log(melspectrogram)
        return log_mel

    if feature_type == FeatureType.STFT:
        stft = librosa.stft(signal)
        stft_db = librosa.amplitude_to_db(stft, ref=np.max)
        return stft_db

    if feature_type == FeatureType.RAW:
        return signal


def get_full_audio_data(feature_types: [FeatureType]):
    data = get_dataset(
        "signal-no-silent-{}-class-dataset-full_audio_sr_22k-greater-than-6-seconds.pkl".format(len(EMOTIONS)))

    np.random.shuffle(data)

    signal_frames = split_audio(data[0]['x'], SR, DURATION)

    while len(signal_frames) < 3:
        np.random.shuffle(data)
        signal_frames = split_audio(data[0]['x'], SR, DURATION)

    x = []
    y_emo = []
    y_gen = []

    for f in signal_frames:
        features = {}

        for feature_type in feature_types:
            feature = _get_feature(feature_type, f)
            features[feature_type.name] = feature

        x.append(features)
        y_emo.append(data[0]['emo'])
        y_gen.append(data[0]['gen'])

    return (np.array(x), np.array(y_emo), np.array(y_gen))
