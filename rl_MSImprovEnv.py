import gym
import numpy as np
import pandas as pd

from Datastore import Datastore
from IMPROVDataset import ImprovDataset
from V4Dataset import V4Datastore
from constants import EMOTIONS, NUM_MFCC, NO_features
from data import FeatureType
from data_versions import DataVersions
from hashing_util import get_hash
from inmemdatastore import InMemDatastore


class ImprovEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, data_version) -> None:
        super().__init__()
        self.itr = 0

        self.X = []
        self.Y = []
        self.num_classes = len(EMOTIONS)

        self.data_version = data_version

        self.datastore = ImprovDataset()

        self.set_data()

        self.action_space = gym.spaces.Discrete(self.num_classes)
        self.observation_space = gym.spaces.Box(-1, 1, [NUM_MFCC, NO_features])

        self.data_hashes = pd.DataFrame(self.datastore.get_data_hash_list())
        self.data_hashes['used'] = False
        self.data_hashes.columns = ['file_hash', 'used']

    def step(self, action):
        assert self.action_space.contains(action)
        reward = -0.1 + int(action == np.argmax(self.Y[self.itr]))
        # reward = 1 if action == self.Y[self.itr] else -1

        done = (len(self.X) - 2 <= self.itr)

        next_state = self.X[self.itr + 1]

        h = get_hash(next_state)
        idx = self.data_hashes.index[self.data_hashes['file_hash'] == h]
        self.data_hashes.at[idx, 'used'] = True

        info = {
            "ground_truth": np.argmax(self.Y[self.itr]),
            "itr": self.itr,
            "used_data_count": int(self.data_hashes[self.data_hashes['used'] == True].shape[0])
        }
        self.itr += 1

        return next_state, reward, done, info

    def render(self, mode='human'):
        print("Not implemented \t i: {}".format(self.itr))

    def reset(self):
        self.itr = 0
        self.set_data()
        return self.X[self.itr]

    def set_data(self):
        self.X = []
        self.Y = []

        (x_train, y_train, y_gen_train), (x_test, y_emo_test, y_gen_test) = self.datastore.get_data()
        # self.X = np.array([d[FeatureType.MFCC.name] for d in x_train])
        assert len(x_train) == len(y_train)
        self.X = x_train
        self.Y = y_train
