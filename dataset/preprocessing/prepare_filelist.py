import os
import glob
import argparse
import torch
from tqdm import tqdm


def filter_audio_len(data_len, wav_min, wav_max):
    return wav_min <= data_len <= wav_max


def replace_path(path, old, new):
    return path.replace(old, new)


def make_filelist(file_list, filename):
    with open(filename, 'w') as file:
        for item in file_list:
            file.write(item + '\n')


def main(a):
    os.makedirs(a.output_dir, exist_ok=True)

    wavs = sorted(glob.glob(a.input_dir + '/**/*.wav', recursive=True))
    print("Wav num: ", len(wavs))

    valid_wavs, short_audio, long_audio = [], 0, 0

    # valid F0
    for wav in tqdm(wavs):
        f0_path = replace_path(wav, 'audio_data', 'feature_data/f0_data').replace('.wav', '.pt')
        f0_value = torch.load(f0_path)
        if f0_value.sum() != 0:
            valid_wavs.append(wav) 

    print(f"wav num: {len(wavs)}")
    print(f"valid F0 num: {len(valid_wavs)}")
    print(f"short wav num: {short_audio}")
    print(f"long wav num: {long_audio}")

    make_filelist(wavs, f'{a.output_dir}/train_wav.txt')

    for i in ['f0', 'w2v']:
        filtered = [replace_path(wav, 'audio_data', f'feature_data/{i}_data').replace('.wav', '.pt') for wav in wavs]
        make_filelist(filtered, f'{a.output_dir}/train_{i}.txt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', default='/workspace/raid/dataset/LibriTTS_16k/train-clean-100')
    parser.add_argument('-o', '--output_dir', default='/workspace/ha0/data_preprocess/filelist')   
    a = parser.parse_args()
    main(a)
