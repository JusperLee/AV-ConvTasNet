'''
Author: Kai Li
Date: 2021-03-11 15:35:11
LastEditors: Kai Li
LastEditTime: 2021-04-06 20:45:23
'''
import argparse
import json
import os
import soundfile as sf
from tqdm import tqdm

def get_mouth_path(in_mouth_dir, wav_file, out_filename, data_type):
    wav_file = wav_file.split('_')
    if out_filename == 's1':
        file_path = os.path.join(in_mouth_dir, data_type, '{}_{}.npz'.format(wav_file[0], wav_file[1]))
    else:
        file_path = os.path.join(in_mouth_dir, data_type, '{}_{}.npz'.format(wav_file[3], wav_file[4]))
    return file_path

def preprocess_one_dir(in_audio_dir, in_mouth_dir, out_dir, data_type):
    """ Create .json file for one condition."""
    file_infos = []
    in_audio_dir = os.path.abspath(in_audio_dir+'/mix')
    wav_list = os.listdir(in_audio_dir)
    wav_list.sort()
    for wav_file in tqdm(wav_list):
        if not wav_file.endswith(".wav"):
            continue
        wav_path = os.path.join(in_audio_dir, wav_file)
        samples = sf.SoundFile(wav_path)
        file_infos.append((wav_path, wav_path.replace('mix', 's1'), get_mouth_path(in_mouth_dir, wav_file, 's1', data_type),len(samples)))
        file_infos.append((wav_path, wav_path.replace('mix', 's2'), get_mouth_path(in_mouth_dir, wav_file, 's2', data_type), len(samples)))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(os.path.join(out_dir + ".json"), "w") as f:
        json.dump(file_infos, f, indent=4)


def preprocess(inp_args):
    """ Create .json files for all conditions."""
    speaker_list = ["mix"] + [f"s{n+1}" for n in range(inp_args.n_src)]
    for data_type in ["tr", "cv", "tt"]:
        preprocess_one_dir(
                os.path.join(inp_args.in_audio_dir, data_type),
                inp_args.in_mouth_dir,
                os.path.join(inp_args.out_dir, data_type),
                data_type
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("WSJ0-MIX data preprocessing")
    parser.add_argument(
        "--in_audio_dir", type=str, default=None, help="Directory path of wham including tr, cv and tt"
    )
    parser.add_argument(
        "--in_mouth_dir", type=str, default=None, help="Directory path to put input lip files"
    )
    parser.add_argument("--n_src", type=int, default=2, help="Number of sources in wsj0-mix")
    parser.add_argument(
        "--out_dir", type=str, default=None, help="Directory path to put output files"
    )
    args = parser.parse_args()
    print(args)
    preprocess(args)