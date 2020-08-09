import librosa
import parselmouth

from constants import SR

if __name__ == '__main__':
    filename = "/Volumes/Kingston/datasets/audio/iemocap/Session1/sentences/wav/Ses01F_script01_3/Ses01F_script01_3_M030_processed.wav"

    audio, sr = librosa.load(filename, sr=SR)

    snd = parselmouth.Sound(audio, sampling_frequency=SR)
    pitch = snd.to_pitch(method=parselmouth.Sound.ToPitchMethod)
    frequencies = pitch.selected_array['frequency']
