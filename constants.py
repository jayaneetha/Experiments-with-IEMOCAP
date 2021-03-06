import socket

host = socket.gethostname()

# environment specific constants

DATA_ROOT = '/iemocap'
PKL_ROOT = '/iemocap/pkl/'

if host == 'asimov':
    DATA_ROOT = '/data/aq/shared/iemocap/IEMOCAP_full_release'
    PKL_ROOT = '/home/u1116888/projects/iemocap_dataset/'

if host == 'Thejans-MacBook-Pro.local':
    DATA_ROOT = '/Volumes/Kingston/datasets/audio/iemocap'
    PKL_ROOT = '/Users/jayaneetha/PycharmProjects/iemocap_dataset/pkl/'

if host == 'thejanr-u20dt':
    DATA_ROOT = '/home/jayaneetha/iotheath/data/iemocap'
    PKL_ROOT = '/home/jayaneetha/iotheath/iemocap_dataset/'

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

# SAVEE CONSTANTS
SAVEE_SR = 44100
SAVEE_DURATION = 3
SAVEE_NO_features = 259

# IMPROV CONSTANTS
# IMPROV_SR = 44100
IMPROV_SR = 22050
IMPROV_DURATION = 2
# IMPROV_NO_features = 173
IMPROV_NO_features = 87
