import numpy as np

from Datastore import Datastore
from Framework import get_dataset, randomize_split
from constants import EMOTIONS
from data import FeatureType, _get_feature


class V4Datastore(Datastore):
    data_pkl = None
    data = []

    def __init__(self, feature_type: FeatureType) -> None:
        self.data_pkl = get_dataset("signal-no-silent-{}-class-dataset-2sec_sr_22k.pkl".format(len(EMOTIONS)))
        for d in self.data_pkl:
            single_file = {}
            feature = _get_feature(feature_type, d['x'])
            single_file[feature_type.name] = feature
            single_file['y_emo'] = d['emo']
            single_file['y_gen'] = d['gen']
            self.data.append(single_file)

    def get_data(self):
        training_data, testing_data = randomize_split(self.data)

        x_train_mfcc = np.array([d[FeatureType.MFCC.name] for d in training_data])
        y_train_emo = np.array([d['y_emo'] for d in training_data])
        y_train_gen = np.array([d['y_gen'] for d in training_data])

        x_test_mfcc = np.array([d[FeatureType.MFCC.name] for d in testing_data])
        y_test_emo = np.array([d['y_emo'] for d in training_data])
        y_test_gen = np.array([d['y_gen'] for d in training_data])

        return (x_train_mfcc, y_train_emo, y_train_gen), (x_test_mfcc, y_test_emo, y_test_gen)
