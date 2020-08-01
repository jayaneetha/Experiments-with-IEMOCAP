import librosa
import matplotlib.pyplot as plt
import numpy as np

from legacy.Framework import get_dataset

data = get_dataset('signal-dataset.pkl')
melspectrogram = librosa.feature.melspectrogram(data[0]['x'], 22050)
log_mel = np.log(melspectrogram)
plt.figure()
plt.imshow(log_mel)
plt.show()
