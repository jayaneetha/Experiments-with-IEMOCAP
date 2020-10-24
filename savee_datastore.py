import numpy as np

from Datastore import Datastore
from Framework import get_dataset
from constants import EMOTIONS
from data import FeatureType
from hashing_util import get_hash


class SAVEEDatastore(Datastore):
    data_pkl = None

    def __init__(self, feature_type: FeatureType):
        if not (FeatureType.MFCC == feature_type):
            raise Exception("Only supports {}".format(FeatureType.MFCC.name))

        self.data_pkl = get_dataset("savee_sr_44k_3sec_{}-classes.pkl".format(len(EMOTIONS)))

    def get_data(self):
        np.random.shuffle(self.data_pkl)
        x_train_mfcc = np.array([d[FeatureType.MFCC.name] for d in self.data_pkl])
        y_train_emo = np.array([d['y_emo'] for d in self.data_pkl])

        return (x_train_mfcc, y_train_emo, None), (None, None, None)

    def get_data_hash_list(self):
        mfccs = np.array([d[FeatureType.MFCC.name] for d in self.data_pkl])
        hashes = []
        for m in mfccs:
            hashes.append(get_hash(m))
        return np.array(hashes)
