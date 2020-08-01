import glob
import pickle

import librosa
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras.utils import to_categorical

from constants import DATA_ROOT, EMOTIONS, GENDERS, DURATION, NUM_MFCC, SR

dataset_base_path = DATA_ROOT


def get_emotion_of_sentence(session_id, dialog_id, sentence_id):
    emo_evaluation_file = dataset_base_path + '/' + session_id + '/dialog/EmoEvaluation/' + dialog_id + '.txt'
    with open(emo_evaluation_file, 'r') as f:
        targets = [line for line in f if sentence_id in line]
        emo = targets[0].split('\t')[2]
        if emo in EMOTIONS:
            return EMOTIONS.index(emo)
        else:
            return -1


def get_gender_of_sentence(sentence_id):
    sess = sentence_id.split("_")[-1]
    return GENDERS.index(sess[0])


def get_files_list(session_id, dialog_id='*'):
    sentences_dir = dataset_base_path + '/' + session_id + '/sentences/wav/' + dialog_id
    file_list = glob.glob(sentences_dir + "/*.wav")
    return file_list


def load_wav(filename, sr=None):
    audio, sr = librosa.load(filename, sr=sr)
    return audio, sr


def split_audio(signal, sr, split_duration):
    length = split_duration * sr

    if length < len(signal):
        frames = librosa.util.frame(signal, frame_length=length, hop_length=length).T
        return frames
    else:
        audio = add_missing_padding(signal, sr, split_duration)
        frames = [audio]
        return np.array(frames)


def remove_silent(signal, top_db=25):
    split_times = librosa.effects.split(signal, top_db=top_db)
    mix = []
    for s in split_times:
        part = signal[s[0]:s[1]]
        mix.extend(part)

    return np.array(mix)


def add_missing_padding(audio, sr, duration):
    signal_length = duration * sr
    audio_length = audio.shape[0]
    padding_length = signal_length - audio_length
    if padding_length > 0:
        padding = np.zeros(padding_length)
        signal = np.hstack((audio, padding))
        return signal
    return audio


def get_mfcc(filename, duration):
    audio, sr = load_wav(filename)
    signal = add_missing_padding(audio, sr, duration)
    return _get_mfcc(signal, sr)


def _get_mfcc(signal, sr):
    return librosa.feature.mfcc(signal, sr, n_mfcc=NUM_MFCC)


def get_rms(filename, duration):
    audio, sr = load_wav(filename)
    signal = add_missing_padding(audio, sr, duration)
    return _get_rms(signal)


def _get_rms(signal):
    return librosa.feature.rms(y=signal)


def show_mfcc(filename):
    mfccs = get_mfcc(filename, DURATION)

    plt.figure(figsize=(10, 4))
    plt.imshow(mfccs)
    plt.colorbar()
    plt.title('MFCC | ' + str(mfccs.shape))
    plt.tight_layout()
    plt.show()


def get_details_of_path(path):
    sections = path.split('/')
    details = {
        'SESSION_ID': sections[-5],
        'DIALOG_ID': sections[-2],
        'SENTENCE_ID': sections[-1].split(".")[0]
    }
    return details


def get_dataset(session_id='*', sampling_rate=None):
    files = get_files_list(session_id)

    details = []
    X = []
    Y_emo = []
    Y_gen = []

    for f in files:
        file_details = get_details_of_path(f)
        emo = get_emotion_of_sentence(file_details['SESSION_ID'], file_details['DIALOG_ID'],
                                      file_details['SENTENCE_ID'])
        gen = get_gender_of_sentence(file_details['SENTENCE_ID'])

        if emo > -1:
            audio, sr = load_wav(f, sr=sampling_rate)

            audio = remove_silent(audio)

            signal_frames = split_audio(audio, sr, DURATION)

            for frame in signal_frames:
                X.append(frame)
                Y_emo.append(emo)
                Y_gen.append(gen)
                details.append(file_details)

    return np.array(X), to_categorical(Y_emo, num_classes=len(EMOTIONS)), to_categorical(Y_gen, num_classes=len(
        GENDERS)), np.array(details)


if __name__ == "__main__":
    print("Start")

    x, ye, yg, d = get_dataset(sampling_rate=SR)

    print("loaded dataset")

    data = []

    max_count_per_emo_class = 1200
    dataset_counts = np.zeros(len(EMOTIONS))

    print("re-structuring to store")
    for i in range(len(x)):
        e = np.argmax(ye[i])
        if max_count_per_emo_class > 0:
            if dataset_counts[e] < max_count_per_emo_class:
                data.append({
                    'x': x[i],
                    'emo': ye[i],
                    'gen': yg[i],
                    'detail': d[i]
                })
                dataset_counts[e] = dataset_counts[e] + 1
        else:
            data.append({
                'x': x[i],
                'emo': ye[i],
                'gen': yg[i],
                'detail': d[i]
            })
            dataset_counts[e] = dataset_counts[e] + 1

    print(dataset_counts)
    print("{} records".format(len(data)))
    data = np.array(data)

    print("restructured. \n saving...")

    with open('signal-no-silent-{}-class-dataset-{}sec_sr_16k.pkl'.format(len(EMOTIONS), DURATION), 'wb') as f:
        pickle.dump(data, f)

    print('end')
