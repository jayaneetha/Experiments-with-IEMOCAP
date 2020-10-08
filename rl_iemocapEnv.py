import gym
import numpy as np
from enum import Enum

from Datastore import Datastore
from V4Dataset import V4Datastore
from constants import EMOTIONS, NUM_MFCC, NO_features
from data import FeatureType
from inmemdatastore import InMemDatastore


class DataVersions(Enum):
    V3 = 0,  # ONE-AUDIO-ONE-EPISODE
    V4 = 1


class IEMOCAPEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, data_version) -> None:
        super().__init__()
        self.itr = 0

        self.X = []
        self.Y = []
        self.num_classes = len(EMOTIONS)

        self.data_version = data_version

        self.datastore: Datastore

        if data_version == DataVersions.V3:
            self.datastore = InMemDatastore(FeatureType.MFCC)

        if data_version == DataVersions.V4:
            self.datastore = V4Datastore(FeatureType.MFCC)

        self.set_data()

        self.action_space = gym.spaces.Discrete(self.num_classes)
        self.observation_space = gym.spaces.Box(-1, 1, [NUM_MFCC, NO_features])

    def step(self, action):
        assert self.action_space.contains(action)
        reward = -0.1 + int(action == np.argmax(self.Y[self.itr]))
        # reward = 1 if action == self.Y[self.itr] else -1

        done = (len(self.X) - 2 <= self.itr)

        next_state = self.X[self.itr + 1]
        info = {
            "ground_truth": np.argmax(self.Y[self.itr]),
            "itr": self.itr
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

        if self.data_version == DataVersions.V4:
            (x_train, y_train, y_gen_train), (x_test, y_emo_test, y_gen_test) = self.datastore.get_data()
            # self.X = np.array([d[FeatureType.MFCC.name] for d in x_train])
            assert len(x_train) == len(y_train)
            self.X = x_train
            self.Y = y_train

        if self.data_version == DataVersions.V3:
            (x_train, y_train, y_gen_train) = self.datastore.get_data()
            assert len(x_train) == len(y_train)
            self.X = np.array([d[FeatureType.MFCC.name] for d in x_train])
            self.Y = y_train
