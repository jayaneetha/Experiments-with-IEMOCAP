import pickle

import numpy as np
from sklearn import svm

from legacy.convert_iemocap_dataset_to_pkl import get_files_list, get_details_of_path, get_emotion_of_sentence, \
    load_wav, \
    add_missing_padding

LOAD_FROM_PKL = True

X = []
Y_emo = []

if LOAD_FROM_PKL:

    with open('SVM-X.pkl', 'rb') as f:
        X = pickle.load(f)

    with open('SVM-y_emo.pkl', 'rb') as f:
        Y_emo = pickle.load(f)
else:

    files = get_files_list('*')

    print("files", len(files))

    for f in files:
        file_details = get_details_of_path(f)
        emo = get_emotion_of_sentence(file_details['SESSION_ID'], file_details['DIALOG_ID'],
                                      file_details['SENTENCE_ID'])

        audio, sr = load_wav(f)
        signal = add_missing_padding(audio, sr)
        X.append(signal)

        Y_emo.append(emo)

print("loaded dataset")

data = []

print("re-structuring to store")
for i in range(len(X)):
    data.append({
        'x': X[i],
        'emo': Y_emo[i]
    })

print("{} records".format(len(data)))
data = np.array(data)

# shuffle the dataset
np.random.shuffle(data)

# divide training and testing dataset
training_count = int(len(data) * .8)

training_data = data[:training_count]
testing_data = data[training_count:]

x_train, y_emo_train = [], []

print("creating x_train")
for d in training_data:
    x_train.append(d['x'])
    y_emo_train.append(d['emo'])

x_train = np.array(x_train)
y_emo_train = np.array(y_emo_train)

print("fitting")
clf = svm.SVC(verbose=1)
clf.fit(X=x_train, y=y_emo_train)

print("fitting end")

print("creating x_test")
x_test, y_emo_test = [], []

for d in testing_data:
    x_test.append(d['x'])
    y_emo_test.append(d['emo'])

x_test = np.array(x_test)
y_emo_test = np.array(y_emo_test)

print("Evaluating")
print(clf.score(X=x_test, y=y_emo_test))

print("end")
