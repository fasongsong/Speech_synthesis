import random
import numpy as np
import torch
import torch.utils.data
from torch.autograd import Variable
import layers
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence
import SpecAugment

class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text, hparams, warp_set="og"):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.n_speakers = hparams.speaker_num
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        # self.mel_time_warping = hparams.mel_time_warping
        # self.mel_time_length_adjustment = hparams.mel_time_length_adjustment
        # self.mel_time_length_adjustment_double = hparams.mel_time_length_adjustment_double
        # self.mel_time_mask = hparams.mel_time_mask
        # self.mel_freq_mask = hparams.mel_freq_mask
        # self.mel_freq_warping = hparams.mel_freq_warping
        self.value_adjustmet = hparams.value_adjustmet
        self.stft = layers.TacotronSTFT(hparams)
        self.warp_set = warp_set
        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)

    def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text, speaker_id = audiopath_and_text[0], audiopath_and_text[1], audiopath_and_text[2]
        text = self.get_text(text) # int_tensor[char_index, ....]
        mel = self.get_mel(audiopath) # []
        speaker_id = self.get_speaker(speaker_id)

        if self.warp_set == "og":
            aug_label = torch.LongTensor([1,0])
            warp_label = torch.LongTensor([1, 0, 0])
            return (text, mel, speaker_id, aug_label, warp_label)

        elif self.warp_set == "time":
            time_warping_mel = SpecAugment.local_random_time_warping(mel, 0.3)
            aug_label = torch.LongTensor([0,1])
            warp_label = torch.LongTensor([0, 1, 0])
            return (text, time_warping_mel, speaker_id, aug_label, warp_label)

        elif self.warp_set == "freq":
            freq_warping_mel = SpecAugment.local_random_freq_warping(mel, 4.0)
            aug_label = torch.LongTensor([0,1])
            warp_label = torch.LongTensor([0, 0, 1])
            return (text, freq_warping_mel, speaker_id, aug_label, warp_label)

        return (original, time_warp, freq_warp)

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return melspec

    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
        return text_norm

    def get_speaker(self, speaker):
        speaker_vector = np.zeros(self.n_speakers)
        speaker_vector[int(speaker)] = 1
        return torch.Tensor(speaker_vector.astype(dtype=np.float32))


    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [[text_normalized, mel_normalized], ...]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        speakers = torch.LongTensor(len(batch), len(batch[0][2]))
        for i in range(len(ids_sorted_decreasing)):
            speaker = batch[ids_sorted_decreasing[i]][2]
            speakers[i, :] = speaker

        aug_labels = torch.LongTensor(len(batch), len(batch[0][3]))
        for i in range(len(ids_sorted_decreasing)):
            label1 = batch[ids_sorted_decreasing[i]][3]
            aug_labels[i, :] = label1

        warp_labels = torch.LongTensor(len(batch), len(batch[0][4]))
        for i in range(len(ids_sorted_decreasing)):
            label2 = batch[ids_sorted_decreasing[i]][4]
            warp_labels[i, :] = label2

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)

        return text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths, speakers, aug_labels, warp_labels

# import random
# import numpy as np
# import torch
# import torch.utils.data
#
# import layers
# from utils import load_wav_to_torch, load_filepaths_and_text
# from text import text_to_sequence
#
#
# class TextMelLoader(torch.utils.data.Dataset):
#     """
#         1) loads audio,text pairs
#         2) normalizes text and converts them to sequences of one-hot vectors
#         3) computes mel-spectrograms from audio files.
#     """
#     def __init__(self, audiopaths_and_text, hparams):
#         self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
#         self.text_cleaners = hparams.text_cleaners
#         self.max_wav_value = hparams.max_wav_value
#         self.sampling_rate = hparams.sampling_rate
#         self.load_mel_from_disk = hparams.load_mel_from_disk
#         self.stft = layers.TacotronSTFT(
#             hparams.filter_length, hparams.hop_length, hparams.win_length,
#             hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
#             hparams.mel_fmax)
#         random.seed(1234)
#         random.shuffle(self.audiopaths_and_text)
#
#     def get_mel_text_pair(self, audiopath_and_text):
#         # separate filename and text
#         audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
#         text = self.get_text(text) # int_tensor[char_index, ....]
#         mel = self.get_mel(audiopath) # []
#         return (text, mel)
#
#     def get_mel(self, filename):
#         if not self.load_mel_from_disk:
#             audio, sampling_rate = load_wav_to_torch(filename)
#             if sampling_rate != self.stft.sampling_rate:
#                 raise ValueError("{} {} SR doesn't match target {} SR".format(
#                     sampling_rate, self.stft.sampling_rate))
#             audio_norm = audio / self.max_wav_value
#             audio_norm = audio_norm.unsqueeze(0)
#             audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
#             melspec = self.stft.mel_spectrogram(audio_norm)
#             melspec = torch.squeeze(melspec, 0)
#         else:
#             melspec = torch.from_numpy(np.load(filename))
#             assert melspec.size(0) == self.stft.n_mel_channels, (
#                 'Mel dimension mismatch: given {}, expected {}'.format(
#                     melspec.size(0), self.stft.n_mel_channels))
#
#         return melspec
#
#     def get_text(self, text):
#         text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
#         return text_norm
#
#     def __getitem__(self, index):
#         return self.get_mel_text_pair(self.audiopaths_and_text[index])
#
#     def __len__(self):
#         return len(self.audiopaths_and_text)
#
#
# class TextMelCollate():
#     """ Zero-pads model inputs and targets based on number of frames per setep
#     """
#     def __init__(self, n_frames_per_step):
#         self.n_frames_per_step = n_frames_per_step
#
#     def __call__(self, batch):
#         """Collate's training batch from normalized text and mel-spectrogram
#         PARAMS
#         ------
#         batch: [[text_normalized, mel_normalized], ...]
#         """
#         # Right zero-pad all one-hot text sequences to max input length
#         input_lengths, ids_sorted_decreasing = torch.sort(
#             torch.LongTensor([len(x[0]) for x in batch]),
#             dim=0, descending=True)
#         max_input_len = input_lengths[0]
#
#         text_padded = torch.LongTensor(len(batch), max_input_len)
#         text_padded.zero_()
#         for i in range(len(ids_sorted_decreasing)):
#             text = batch[ids_sorted_decreasing[i]][0]
#             text_padded[i, :text.size(0)] = text
#
#         # Right zero-pad mel-spec
#         num_mels = batch[0][1].size(0)
#         max_target_len = max([x[1].size(1) for x in batch])
#         if max_target_len % self.n_frames_per_step != 0:
#             max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
#             assert max_target_len % self.n_frames_per_step == 0
#
#         # include mel padded and gate padded
#         mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
#         mel_padded.zero_()
#         gate_padded = torch.FloatTensor(len(batch), max_target_len)
#         gate_padded.zero_()
#         output_lengths = torch.LongTensor(len(batch))
#         for i in range(len(ids_sorted_decreasing)):
#             mel = batch[ids_sorted_decreasing[i]][1]
#             mel_padded[i, :, :mel.size(1)] = mel
#             gate_padded[i, mel.size(1)-1:] = 1
#             output_lengths[i] = mel.size(1)
#
#         return text_padded, input_lengths, mel_padded, gate_padded, \
#             output_lengths
