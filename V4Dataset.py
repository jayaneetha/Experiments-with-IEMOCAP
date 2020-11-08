import numpy as np
import pandas as pd

from Datastore import Datastore
from Framework import get_dataset, randomize_split
from constants import EMOTIONS
from data import FeatureType, _get_feature
from hashing_util import get_hash

FILTER_DATA = False


class V4Datastore(Datastore):
    data_pkl = None
    data = []
    pre_train_data = []

    def __init__(self, feature_type: FeatureType) -> None:
        self.data_pkl = get_dataset("signal-no-silent-{}-class-dataset-2sec_sr_22k.pkl".format(len(EMOTIONS)))
        for d in self.data_pkl:
            single_file = {}
            feature = _get_feature(feature_type, d['x'])
            single_file[feature_type.name] = feature
            single_file['y_emo'] = d['emo']
            single_file['y_gen'] = d['gen']
            self.data.append(single_file)

        rl_data, pre_train_data = randomize_split(self.data, split_ratio=0.7)

        self.data = rl_data
        self.pre_train_data = pre_train_data

        self.data_hashes = pd.DataFrame(self.get_data_hash_list())
        self.data_hashes['usage'] = 0
        self.data_hashes.columns = ['file_hash', 'usage']

    def get_data(self):
        training_data, testing_data = randomize_split(self.data)

        x_train_mfcc = np.array([d[FeatureType.MFCC.name] for d in training_data])
        y_train_emo = np.array([d['y_emo'] for d in training_data])
        y_train_gen = np.array([d['y_gen'] for d in training_data])

        x_test_mfcc = np.array([d[FeatureType.MFCC.name] for d in testing_data])
        y_test_emo = np.array([d['y_emo'] for d in training_data])
        y_test_gen = np.array([d['y_gen'] for d in training_data])

        if FILTER_DATA:
            mfccs = np.array([d[FeatureType.MFCC.name] for d in training_data])
            for m in mfccs:
                h = get_hash(m)
                idx = self.data_hashes.index[self.data_hashes['file_hash'] == h]
                self.data_hashes.at[idx, 'usage'] = int(
                    self.data_hashes[self.data_hashes['file_hash'] == h]['usage']) + 1

            self.filter_data()
            del mfccs

        return (x_train_mfcc, y_train_emo, y_train_gen), (x_test_mfcc, y_test_emo, y_test_gen)

    def get_pre_train_data(self):

        training_data = self.pre_train_data

        x_train_mfcc = np.array([d[FeatureType.MFCC.name] for d in training_data])
        y_train_emo = np.array([d['y_emo'] for d in training_data])
        y_train_gen = np.array([d['y_gen'] for d in training_data])

        return x_train_mfcc, y_train_emo, y_train_gen

    def get_data_hash_list(self):
        hashes = []
        mfccs = np.array([d[FeatureType.MFCC.name] for d in self.data])
        for m in mfccs:
            hashes.append(get_hash(m))
        return np.array(hashes)

    def filter_data(self):
        filtered_data = []
        for d in self.data:
            mfcc = d[FeatureType.MFCC.name]
            h = get_hash(mfcc)
            used_count = int(self.data_hashes[self.data_hashes['file_hash'] == h]['usage'])
            if used_count < 100:
                filtered_data.append(d)
        self.data = filtered_data
        if len(self.data) < 5:
            import sys
            sys.exit("Exiting due to lack of training data")
        del filtered_data
