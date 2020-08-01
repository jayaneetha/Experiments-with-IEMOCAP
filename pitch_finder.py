import librosa

from data import get_data, FeatureType

if __name__ == '__main__':
    (x_train, y_emo_train, y_gen_train), (x_test, y_emo_test, y_gen_test) = get_data(feature_type=FeatureType.RAW)

    x = x_train[0]
    pitches, magnitudes = librosa.piptrack(y=x, sr=22050)
    print(pitches)
