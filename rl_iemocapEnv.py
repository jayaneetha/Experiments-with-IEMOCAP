import gym
import numpy as np

from constants import EMOTIONS, NUM_MFCC, NO_features
from data import get_data, FeatureType


class IEMOCAPEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self) -> None:
        super().__init__()
        self.itr = 0

        self.X = []
        self.Y = []
        self.num_classes = len(EMOTIONS)

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

        (x_train, y_train, y_gen_train), (x_test, y_emo_test, y_gen_test) = get_data(feature_types=[FeatureType.MFCC])

        self.X = np.array([d[FeatureType.MFCC.name] for d in x_train])
        self.Y = y_train
