import librosa

from legacy.convert_iemocap_dataset_to_pkl import get_files_list


def main():
    print("Start")
    files = get_files_list('Session1')
    data = {}
    for f in files:
        audio, sr = librosa.load(f)
        data[f] = len(audio) / sr

    print(data)
    with open('lengths_of_audio.txt', 'w') as f:
        print(data, file=f)


if __name__ == '__main__':
    main()
