import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
import argparse
import os

import numpy as np
import time
import torch

from hparams import create_hparams
from layers import TacotronSTFT
import layers
from audio_processing import griffin_lim, mel_denormalize
from utils import load_wav_to_torch
from train import load_model
from text import text_to_sequence
from scipy.io.wavfile import write
from scipy import signal

def get_mel(hparams, filename):

    audio, sampling_rate = load_wav_to_torch(filename)
    audio_norm = audio / hparams.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec = layers.TacotronSTFT(hparams).mel_spectrogram(audio_norm)
    melspec = torch.squeeze(melspec, 0)
    length = torch.LongTensor(1)
    length = melspec.size(1)
    return melspec, length

def plot_data(data, index, output_dir="", figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom',
                        interpolation='none')
    plt.savefig(os.path.join(output_dir, 'sentence_{}.png'.format(index)))

def generate_mels(hparams, checkpoint_path, sentences, speaker_id, trans_con, condition, logvar, cleaner, removing_silence_mel_padding, adding_silence_mel_padding, is_GL, output_dir=""):
    model = load_model(hparams)
    try:
        model = model.module
    except:
        pass
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(checkpoint_path)['state_dict'].items()})

    _ = model.eval()

    speaker_name = hparams.speaker_list[speaker_id]
    audio_path = os.path.join("./wavfile", speaker_name, speaker_name + "_t01_s03.wav")
    refence_mel, ref_length = get_mel(hparams, audio_path)

    output_mels = []
    for i, s in enumerate(sentences):
        sequence = np.array(text_to_sequence(s, cleaner))[None, :]
        sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()

        stime = time.time()
        ### condition_on_ref : (False) Transfer   (Truee) control
        ### condition :   (0) : time,  (1) : freq
        ### logvar : how many
        if trans_con == 0:
            ref = False
        elif trans_con == 1:
            ref = True

        _, mel_outputs_postnet, _, alignments = model.inference(sequence, refence_mel, ref_length, condition_on_ref=ref, condition=condition, logvar=logvar)

        mel = mel_outputs_postnet.data.cpu().numpy()[0][:,:-removing_silence_mel_padding]
        mel = np.append(mel, np.ones((80, adding_silence_mel_padding),dtype=np.float32)*-4.0, axis=1)
        if(is_GL):
            plot_data((mel,
                   alignments.data.cpu().numpy()[0].T), i, output_dir)
        inf_time = time.time() - stime
        print("{}th sentence, Infenrece time: {:.2f}s, len_mel: {}".format(i, inf_time, mel_outputs_postnet.size(2)))
        output_mels.append(mel)
    return output_mels


def mels_to_wavs_GL(hparams, mels, taco_stft, output_dir="", ref_level_db = 0, magnitude_power=1.5):
    for i, mel in enumerate(mels):
        stime = time.time()
        mel_decompress = mel_denormalize(torch.from_numpy(mel).cuda().unsqueeze(0))
        mel_decompress = taco_stft.spectral_de_normalize(mel_decompress + ref_level_db) ** (1/magnitude_power)
        mel_decompress_ = mel_decompress.transpose(1, 2).data.cpu()
        spec_from_mel_scaling = 1000
        spec_from_mel = torch.mm(mel_decompress_[0], taco_stft.mel_basis)
        spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
        spec_from_mel = spec_from_mel * spec_from_mel_scaling
        waveform = griffin_lim(torch.autograd.Variable(spec_from_mel[:, :, :]),
                               taco_stft.stft_fn, 60)
       # waveform = signal.lfilter([1], [1, -0.97], waveform)
        #waveform = torch.as_tensor(waveform[0]).data.cpu().numpy()
        waveform = waveform[0].data.cpu().numpy()
        dec_time = time.time() - stime
        len_audio = float(len(waveform)) / float(hparams.sampling_rate)
        str = "{}th sentence, audio length: {:.2f} sec,  mel_to_wave time: {:.2f}".format(i, len_audio, dec_time)
        print(str)
        write(os.path.join(output_dir, "sentence_{}.wav".format(i)), hparams.sampling_rate, waveform)


def run(hparams, output_dir, checkpoint_path, sentence_path, speaker_id, trans_con, condition, logvar, clenaer, removing_silence_mel_padding,
        adding_silence_mel_padding, is_GL, is_melout, is_metaout):
    f = open(sentence_path, mode='r', encoding='utf-8-sig')
    sentences = [x.strip() for x in f.readlines()]
    print('All sentences to infer:', sentences)
    f.close()
    os.makedirs(output_dir, exist_ok=True)

    stft = TacotronSTFT(hparams)

    mels = generate_mels(hparams, checkpoint_path, sentences, speaker_id, trans_con, condition, logvar, clenaer,
                         removing_silence_mel_padding, adding_silence_mel_padding, is_GL, output_dir,)
    if(is_GL): mels_to_wavs_GL(hparams, mels, stft, output_dir)

    mel_paths = []
    if is_melout:
        mel_dir = os.path.join(output_dir, 'mels')
        os.makedirs(mel_dir, exist_ok=True)

        for i, mel in enumerate(mels):
            mel_path = os.path.join(output_dir, 'mels/', "mel-{}.npy".format(i))
            mel_paths.append(mel_path)
            if(list(mel.shape)[1] >=  hparams.max_decoder_steps - removing_silence_mel_padding):
                continue
            np.save(mel_path, mel)


    if is_metaout:
        with open(os.path.join(output_dir, 'metadata.csv'), 'w', encoding='utf-8') as file:
            lines = []
            for i, s in enumerate(sentences):
                mel_path = mel_paths[i]
                if (list(mels[i].shape)[1] >= hparams.max_decoder_steps - removing_silence_mel_padding):
                    continue
                lines.append('{}|{}\n'.format(mel_path,s))
            file.writelines(lines)

if __name__ == '__main__':
    """
    usage
    python inference.py -o=synthesis/80000 -c=nam_h_ep8/checkpoint_80000 -s=test.txt --silence_mel_padding=3 --is_GL 
        -> wave, figure
    python inference.py -o=kss_mels_given_park_text -c=kakao_kss_model_checkpoint_23500 -s=skip_review_percentile_metadata_n.csv --silence_mel_padding=3 --is_melout --is_metaout 
        -> mels, metadata.csv
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save wave and fig')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('-s', '--sentence_path', type=str, default=None,
                        required=False, help='sentence path')
    parser.add_argument('-i', '--speaker_id', type=int, default=0,
                        required=False, help='speaker_id')
    parser.add_argument('-trans', '--trans_con', type=int, default=0,
                        required=False, help='0 transfer, 1 control')
    parser.add_argument('-con', '--condition', type=int, default=0,
                        required=False, help='condition')
    parser.add_argument('-v', '--logvar', type=int, default=1,
                        required=False, help='condition')
    parser.add_argument('-r', '--ref_audio',  action="store_true", help='Whether to ref_audio or not ')
    parser.add_argument('--removing_silence_mel_padding', type=int, default=1,
                        help='removing existing silence_mel_padding, silence audio size is hop_length * silence mel padding')
    parser.add_argument('--adding_silence_mel_padding', type=int, default=0,
                        help='adding silence_mel_padding, silence audio size is hop_length * silence mel padding')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')
    parser.add_argument('--is_GL', action="store_true", help='Whether to do Giffin & Lim inference or not ')
    parser.add_argument('--is_melout', action="store_true", help='Whether to save melspectrogram file or not ')
    parser.add_argument('--is_metaout', action="store_true", help='Whether to save metadata.csv file for (mel, text) tuple or not ')

    args = parser.parse_args()
    hparams = create_hparams(args.hparams)
    hparams.sampling_rate = 22050
    hparams.filter_length = 1024
    hparams.hop_length = 256
    hparams.win_length = 1024

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    ### condition_on_ref : (False) Transfer   (Truee) control
    ### condition :   (0) : time,  (1) : freq
    ### logvar : how many
    run(hparams, args.output_directory, args.checkpoint_path, args.sentence_path, args.speaker_id, args.trans_con, args.condition, args.logvar, hparams.text_cleaners,
        args.removing_silence_mel_padding, args.adding_silence_mel_padding, args.is_GL, args.is_melout, args.is_metaout)
    # run(hparams, "./test", "./check_aae/checkpoint_8400",  "./test/test.txt", 0, 0, hparams.text_cleaners, 1, 3, True, False, False)

