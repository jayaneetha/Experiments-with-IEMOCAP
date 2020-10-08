import numpy as np
import random

from Datastore import Datastore
from Framework import get_dataset
from constants import EMOTIONS, SR, DURATION
from data import FeatureType, _get_feature
from legacy.convert_iemocap_dataset_to_pkl import split_audio


class InMemDatastore(Datastore):
    def __init__(self, feature_type: FeatureType):
        print("Initializing Datastore")

        self.x = []

        data = get_dataset(
            "signal-no-silent-{}-class-dataset-full_audio_sr_22k-greater-than-6-seconds.pkl".format(len(EMOTIONS)))
        np.random.shuffle(data)

        for d in data:

            single_file = {}

            if len(d['x'] > 6 * SR):
                signal_frames = split_audio(d['x'], SR, DURATION)
                frames = []
                emo = []
                gen = []
                for f in signal_frames:
                    feature = _get_feature(feature_type, f)
                    frames.append({feature_type.name: feature})
                    emo.append(d['emo'])
                    gen.append(d['gen'])

                single_file['frames'] = frames
                single_file['y_emo'] = emo
                single_file['y_gen'] = gen

                self.x.append(single_file)
        print("Initialized Datastore")
        print("Processed {} audio files".format(len(data)))
        print("Used {} audio files".format(len(self.x)))

    def get_data(self):
        idx = random.randint(0, len(self.x) - 1)
        x = np.array(self.x[idx]['frames'])
        y_emo = np.array(self.x[idx]['y_emo'])
        y_gen = np.array(self.x[idx]['y_gen'])
        return x, y_emo, y_gen
