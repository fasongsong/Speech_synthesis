import random
import numpy as np
import torch
from torch.autograd import Variable
import layers
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence
import os
import argparse
from hparams import create_hparams
import csv


####################################################
############# Mel 전용 metafile 생성 ###############
####################################################
def prepare_mel_meta(hparams, audiopath_and_text):

    audiopath_and_texts = load_filepaths_and_text(audiopath_and_text)

    with open(os.path.join('./filelists', 'metadata_mel10_val.csv'), 'w', encoding='utf-8') as csvfile:
        for i in range(len(audiopath_and_texts)):
            audiopath, text, speaker_id = audiopath_and_texts[i][0], audiopath_and_texts[i][1], audiopath_and_texts[i][2]

            out_dir = audiopath[:11]
            file_name = audiopath[12:-4]

            file_path = os.path.join(out_dir, file_name+'.npy')
            wr = csv.writer(csvfile, delimiter='|')
            wr.writerow([file_path, text, speaker_id])
    pass


####################################################
################## Mel .npy 생성 ###################
####################################################
def prepare_mel_npy(hparams, audiopath_and_text):

    audiopath_and_texts = load_filepaths_and_text(audiopath_and_text)

    for i in range(len(audiopath_and_texts)):
        audiopath, text, speaker_id = audiopath_and_texts[i][0], audiopath_and_texts[i][1], audiopath_and_texts[i][2]
        audio, sampling_rate = load_wav_to_torch(audiopath)

        audio_norm = audio / hparams.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = layers.TacotronSTFT(hparams).mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)

        out_dir = audiopath[:11]
        file_name = audiopath[12:-4]

        file = os.path.join(out_dir, file_name)

        np.save(file, melspec)
        print("{} / {}".format(i,len(audiopath_and_texts)))
    pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', type=str,
                        help='directory to save tensorboard logs')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--warm_start', action='store_true',
                        help='load the model only (warm start)')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')

    args = parser.parse_args()
    hparams = create_hparams(args.hparams)

    #prepare_mel_meta(hparams, hparams.training_files)
    prepare_mel_meta(hparams, hparams.validation_files)
    #prepare_mel_npy(hparams, hparams.training_files)
    #prepare_mel_npy(hparams, hparams.validation_files)