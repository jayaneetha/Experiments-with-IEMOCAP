import socket

host = socket.gethostname()

# environment specific constants

DATA_ROOT = '/iemocap'

if host == 'asimov':
    DATA_ROOT = '/data/aq/shared/iemocap/IEMOCAP_full_release'

if host == 'Thejans-MacBook-Pro.local':
    DATA_ROOT = '/Volumes/Kingston/datasets/audio/iemocap'

if host == 'thejanr-u20dt':
    DATA_ROOT = '/home/jayaneetha/iotheath/data/iemocap'

# EMOTIONS = ['neu', 'hap', 'sad', 'ang', 'sur', 'fea', 'dis', 'fru', 'exc', 'oth', 'xxx']
# EMOTIONS = ['neu', 'hap', 'sad', 'ang', 'fru', 'exc']
# EMOTIONS = ['hap', 'sad']
EMOTIONS = ['hap', 'sad', 'ang', 'neu']
GENDERS = ['M', 'F']
NUM_MFCC = 40
# DURATION = 4
DURATION = 2
SR = 22050
NO_features = 87  # sr22050&2sec
# NO_features = 173  # sr22050&4sec
# NO_features = 63  # sr8000&4sec
# NO_features = 63  # sr16000&2sec
# NO_features = 251  # sr16000&8sec

WEEK = str(6)
